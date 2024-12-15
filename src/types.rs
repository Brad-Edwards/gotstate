#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateId(String);

#[derive(Clone, Debug)]
pub struct EventId(String);

impl std::fmt::Display for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
