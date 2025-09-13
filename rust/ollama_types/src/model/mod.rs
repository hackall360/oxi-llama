pub mod capability;
pub mod name;

pub use capability::Capability;
pub use name::{
    default_name, is_valid_namespace, merge, parse_name, parse_name_bare,
    parse_name_from_filepath, Name,
};
