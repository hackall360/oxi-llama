use ollama_types::syncmap::SyncMap;

#[test]
fn syncmap_basic_operations() {
    let map: SyncMap<i32, String> = SyncMap::new();
    map.store(1, "a".to_string());
    assert_eq!(map.load(&1), Some("a".to_string()));
    let items = map.items();
    assert_eq!(items.get(&1).unwrap(), "a");
}
