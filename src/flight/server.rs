//! Ladybug Arrow Flight Server
//!
//! Implements FlightService for MCP-style interactions with BindSpace.

use std::pin::Pin;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::writer::IpcWriteOptions;
use arrow::record_batch::RecordBatch;
use arrow_flight::{
    flight_service_server::FlightService,
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaAsIpc, SchemaResult,
    Ticket, encode::FlightDataEncoderBuilder,
};
use futures::{Stream, StreamExt, TryStreamExt};
use parking_lot::RwLock;
use tonic::{Request, Response, Status, Streaming};

use crate::storage::BindSpace;
use crate::core::Addr;
use crate::search::HdrCascade;

use super::actions::{McpAction, execute_action};

/// Fingerprint schema for Arrow Flight transfers
fn fingerprint_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(1248), false), // 156 * 8 bytes
        Field::new("label", DataType::Utf8, true),
        Field::new("zone", DataType::Utf8, false),
        Field::new("distance", DataType::UInt32, true),
        Field::new("similarity", DataType::Float32, true),
    ]))
}

/// Search result schema for streaming results
fn search_result_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(1248), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("distance", DataType::UInt32, false),
        Field::new("similarity", DataType::Float32, false),
        Field::new("cascade_level", DataType::UInt8, false), // Which HDR level matched
    ]))
}

/// Ladybug Flight Service for MCP interactions
pub struct LadybugFlightService {
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_cascade: Arc<RwLock<HdrCascade>>,
}

impl LadybugFlightService {
    pub fn new(bind_space: Arc<RwLock<BindSpace>>, hdr_cascade: Arc<RwLock<HdrCascade>>) -> Self {
        Self { bind_space, hdr_cascade }
    }
}

type BoxedFlightStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl FlightService for LadybugFlightService {
    type HandshakeStream = BoxedFlightStream<HandshakeResponse>;
    type ListFlightsStream = BoxedFlightStream<FlightInfo>;
    type DoGetStream = BoxedFlightStream<FlightData>;
    type DoPutStream = BoxedFlightStream<PutResult>;
    type DoActionStream = BoxedFlightStream<arrow_flight::Result>;
    type ListActionsStream = BoxedFlightStream<ActionType>;
    type DoExchangeStream = BoxedFlightStream<FlightData>;

    /// Handshake - authenticate client
    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        // Simple passthrough - add auth later
        let output = futures::stream::once(async {
            Ok(HandshakeResponse {
                protocol_version: 0,
                payload: bytes::Bytes::from("ladybug-flight-v1"),
            })
        });
        Ok(Response::new(Box::pin(output)))
    }

    /// List available flights (fingerprint tables)
    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        let schema = fingerprint_schema();
        let options = IpcWriteOptions::default();

        // Surface zone flight
        let surface_info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|e| Status::internal(e.to_string()))?
            .with_descriptor(FlightDescriptor::new_path(vec!["surface".to_string()]))
            .with_total_records(4096) // 16 prefixes * 256 slots
            .with_total_bytes(4096 * 1248);

        // Node zone flight
        let node_info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|e| Status::internal(e.to_string()))?
            .with_descriptor(FlightDescriptor::new_path(vec!["nodes".to_string()]))
            .with_total_records(32768) // 128 prefixes * 256 slots
            .with_total_bytes(32768 * 1248);

        let flights = vec![surface_info, node_info];
        let stream = futures::stream::iter(flights.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    /// Get flight info for a specific table
    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let schema = fingerprint_schema();

        let info = FlightInfo::new()
            .try_with_schema(&schema)
            .map_err(|e| Status::internal(e.to_string()))?
            .with_descriptor(descriptor);

        Ok(Response::new(info))
    }

    /// Poll for flight completion status
    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        Err(Status::unimplemented("poll_flight_info not implemented"))
    }

    /// Get schema for a flight
    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let schema = fingerprint_schema();
        let options = IpcWriteOptions::default();
        let schema_result = SchemaAsIpc::new(&schema, &options)
            .try_into()
            .map_err(|e: arrow::error::ArrowError| Status::internal(e.to_string()))?;
        Ok(Response::new(schema_result))
    }

    /// DoGet - Stream fingerprints or search results
    ///
    /// Ticket format:
    /// - "all" - stream all fingerprints
    /// - "search:<query_hex>:<threshold>" - similarity search with streaming results
    /// - "topk:<query_hex>:<k>" - top-k similar fingerprints
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let ticket_str = String::from_utf8(ticket.ticket.to_vec())
            .map_err(|_| Status::invalid_argument("Invalid ticket encoding"))?;

        let bind_space = self.bind_space.clone();
        let hdr_cascade = self.hdr_cascade.clone();

        if ticket_str == "all" {
            // Stream all fingerprints
            let stream = stream_all_fingerprints(bind_space);
            Ok(Response::new(Box::pin(stream)))
        } else if ticket_str.starts_with("search:") {
            // Similarity search with streaming
            let parts: Vec<&str> = ticket_str.split(':').collect();
            if parts.len() != 3 {
                return Err(Status::invalid_argument("Invalid search ticket format"));
            }
            let query_hex = parts[1];
            let threshold: u32 = parts[2].parse()
                .map_err(|_| Status::invalid_argument("Invalid threshold"))?;

            let query = hex::decode(query_hex)
                .map_err(|_| Status::invalid_argument("Invalid query hex"))?;

            let stream = stream_search_results(bind_space, hdr_cascade, query, threshold);
            Ok(Response::new(Box::pin(stream)))
        } else if ticket_str.starts_with("topk:") {
            let parts: Vec<&str> = ticket_str.split(':').collect();
            if parts.len() != 3 {
                return Err(Status::invalid_argument("Invalid topk ticket format"));
            }
            let query_hex = parts[1];
            let k: usize = parts[2].parse()
                .map_err(|_| Status::invalid_argument("Invalid k"))?;

            let query = hex::decode(query_hex)
                .map_err(|_| Status::invalid_argument("Invalid query hex"))?;

            let stream = stream_topk_results(bind_space, hdr_cascade, query, k);
            Ok(Response::new(Box::pin(stream)))
        } else {
            Err(Status::invalid_argument("Unknown ticket type"))
        }
    }

    /// DoPut - Ingest fingerprints (zero-copy from Arrow buffers)
    async fn do_put(
        &self,
        request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        let mut stream = request.into_inner();
        let bind_space = self.bind_space.clone();

        let mut total_records = 0i64;

        // Process incoming batches
        while let Some(flight_data) = stream.next().await {
            let flight_data = flight_data?;

            // Decode Arrow IPC message
            if let Ok(batch) = decode_flight_data(&flight_data) {
                let count = ingest_batch(&bind_space, &batch)?;
                total_records += count as i64;
            }
        }

        let result = PutResult {
            app_metadata: bytes::Bytes::from(format!("ingested:{}", total_records)),
        };

        let stream = futures::stream::once(async { Ok(result) });
        Ok(Response::new(Box::pin(stream)))
    }

    /// DoAction - Execute MCP tools
    ///
    /// Actions:
    /// - "encode" - Encode text/data to fingerprint (Sigma-10 membrane)
    /// - "bind" - Bind fingerprint to address
    /// - "resonate" - Find similar fingerprints
    /// - "hamming" - Compute Hamming distance
    /// - "xor_bind" - XOR bind two fingerprints
    async fn do_action(
        &self,
        request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        let action = request.into_inner();
        let action_type = action.r#type.as_str();
        let body = action.body;

        let bind_space = self.bind_space.clone();
        let hdr_cascade = self.hdr_cascade.clone();

        let result = execute_action(action_type, &body, bind_space, hdr_cascade)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let flight_result = arrow_flight::Result {
            body: bytes::Bytes::from(result),
        };

        let stream = futures::stream::once(async { Ok(flight_result) });
        Ok(Response::new(Box::pin(stream)))
    }

    /// List available actions (MCP tools)
    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let actions = vec![
            ActionType {
                r#type: "encode".to_string(),
                description: "Encode text/data to 10K-bit fingerprint via Sigma-10 membrane".to_string(),
            },
            ActionType {
                r#type: "bind".to_string(),
                description: "Bind fingerprint to BindSpace address".to_string(),
            },
            ActionType {
                r#type: "resonate".to_string(),
                description: "Find similar fingerprints via HDR cascade search".to_string(),
            },
            ActionType {
                r#type: "hamming".to_string(),
                description: "Compute Hamming distance between two fingerprints".to_string(),
            },
            ActionType {
                r#type: "xor_bind".to_string(),
                description: "XOR bind two fingerprints (holographic composition)".to_string(),
            },
        ];

        let stream = futures::stream::iter(actions.into_iter().map(Ok));
        Ok(Response::new(Box::pin(stream)))
    }

    /// DoExchange - Bidirectional streaming (not implemented)
    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange not implemented"))
    }
}

// =============================================================================
// Helper functions for streaming
// =============================================================================

fn stream_all_fingerprints(
    bind_space: Arc<RwLock<BindSpace>>,
) -> impl Stream<Item = Result<FlightData, Status>> {
    let schema = fingerprint_schema();

    futures::stream::unfold((bind_space, 0u8, schema), |(bs, prefix, schema)| async move {
        if prefix > 0xFF {
            return None;
        }

        // Build batch for this prefix
        let batch = {
            let space = bs.read();
            build_prefix_batch(&space, prefix, &schema)
        };

        match batch {
            Ok(Some(b)) => {
                let encoder = FlightDataEncoderBuilder::new()
                    .with_schema(schema.clone())
                    .build(futures::stream::once(async { Ok(b) }));

                // Collect encoder output
                let flight_data: Vec<FlightData> = encoder
                    .filter_map(|r| async { r.ok() })
                    .collect()
                    .await;

                let next_prefix = prefix.saturating_add(1);
                Some((futures::stream::iter(flight_data.into_iter().map(Ok)), (bs, next_prefix, schema)))
            }
            Ok(None) => {
                let next_prefix = prefix.saturating_add(1);
                Some((futures::stream::iter(vec![]), (bs, next_prefix, schema)))
            }
            Err(e) => {
                Some((futures::stream::iter(vec![Err(e)]), (bs, 0xFF + 1, schema)))
            }
        }
    })
    .flatten()
}

fn stream_search_results(
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_cascade: Arc<RwLock<HdrCascade>>,
    query: Vec<u8>,
    threshold: u32,
) -> impl Stream<Item = Result<FlightData, Status>> {
    // TODO: Implement streaming HDR cascade search
    // For now, return empty stream
    futures::stream::empty()
}

fn stream_topk_results(
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_cascade: Arc<RwLock<HdrCascade>>,
    query: Vec<u8>,
    k: usize,
) -> impl Stream<Item = Result<FlightData, Status>> {
    // TODO: Implement streaming TopK search
    // For now, return empty stream
    futures::stream::empty()
}

fn build_prefix_batch(
    _space: &BindSpace,
    _prefix: u8,
    _schema: &SchemaRef,
) -> Result<Option<RecordBatch>, Status> {
    // TODO: Build RecordBatch from BindSpace prefix
    Ok(None)
}

fn decode_flight_data(_flight_data: &FlightData) -> Result<RecordBatch, Status> {
    // TODO: Decode Arrow IPC from FlightData
    Err(Status::unimplemented("decode not implemented"))
}

fn ingest_batch(
    _bind_space: &Arc<RwLock<BindSpace>>,
    _batch: &RecordBatch,
) -> Result<usize, Status> {
    // TODO: Ingest RecordBatch into BindSpace
    Ok(0)
}
