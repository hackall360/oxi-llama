use regex::Regex;

/// Infer opening and closing tags that surround thinking traces.
pub fn infer_tags(tmpl: &str) -> (String, String) {
    let re = Regex::new(r"(?s)\{\{-?\s*(.*?)\s*-?\}\}").unwrap();
    let mut stack: Vec<String> = Vec::new();
    let mut last_end = 0;
    let mut open_tag = String::new();
    let mut close_tag = String::new();
    for mat in re.find_iter(tmpl) {
        let token = &tmpl[mat.start()..mat.end()];
        let inner = token.trim_start_matches("{{").trim_end_matches("}}");
        let inner = inner.trim_matches('-').trim();
        let before_text = &tmpl[last_end..mat.start()];
        if inner.starts_with("range") {
            if inner.contains(".Messages") {
                stack.push("range_messages".into());
            } else {
                stack.push("range".into());
            }
        } else if inner.starts_with("end") {
            stack.pop();
        } else if inner.starts_with("if") || inner.starts_with("with") {
            stack.push(inner.to_string());
        } else if inner.contains(".Thinking") {
            let mut most_recent_range = None;
            for item in stack.iter().rev() {
                if item.starts_with("range") {
                    most_recent_range = Some(item.clone());
                    break;
                }
            }
            if let Some(r) = most_recent_range {
                if r == "range_messages" {
                    open_tag = before_text.trim().to_string();
                    if let Some(next) = re.find_at(tmpl, mat.end()) {
                        let after = &tmpl[mat.end()..next.start()];
                        close_tag = after.trim().to_string();
                    } else {
                        let after = &tmpl[mat.end()..];
                        close_tag = after.trim().to_string();
                    }
                    break;
                }
            }
        }
        last_end = mat.end();
    }
    (open_tag, close_tag)
}

