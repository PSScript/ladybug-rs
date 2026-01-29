//! World state

pub struct World {
    pub version: u64,
    // Lance storage handle would go here
}

impl World {
    pub fn fork(&self) -> World {
        World {
            version: self.version,
        }
    }
}
