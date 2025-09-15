/// Infer opening and closing tags that surround thinking traces by walking a
/// simplified parse tree of the template.
pub fn infer_tags(tmpl: &str) -> (String, String) {
    let ast = parse_template(tmpl);
    let mut ancestors = Vec::new();
    let mut result: Option<(String, String)> = None;
    traverse_list(&ast, &mut ancestors, &mut result);
    result.unwrap_or_default()
}

#[derive(Debug, Default)]
struct ListNode {
    nodes: Vec<Node>,
}

#[derive(Debug)]
struct BranchNode {
    pipe: String,
    list: ListNode,
    else_list: Option<ListNode>,
}

#[derive(Debug)]
struct ActionNode {
    raw: String,
}

#[derive(Debug)]
enum Node {
    Text(String),
    Action(ActionNode),
    Range(BranchNode),
    If(BranchNode),
    With(BranchNode),
}

#[derive(Debug, Clone, Copy)]
enum BranchKind {
    Range,
    If,
    With,
}

#[derive(Debug)]
enum Token {
    Text(String),
    Action(ActionToken),
}

#[derive(Debug)]
struct ActionToken {
    raw: String,
    kind: ActionKind,
}

#[derive(Debug, Clone)]
enum ActionKind {
    RangeStart { pipe: String },
    IfStart { pipe: String },
    WithStart { pipe: String },
    Else,
    End,
    Comment,
    Other,
}

#[derive(Debug)]
enum Context {
    List {
        nodes: Vec<Node>,
    },
    Branch {
        kind: BranchKind,
        pipe: String,
        main: Vec<Node>,
        else_nodes: Vec<Node>,
        in_else: bool,
    },
}

fn parse_template(input: &str) -> ListNode {
    let tokens = tokenize(input);
    let mut stack = vec![Context::List { nodes: Vec::new() }];
    for token in tokens {
        match token {
            Token::Text(text) => {
                current_nodes_mut(&mut stack).push(Node::Text(text));
            }
            Token::Action(action) => match action.kind {
                ActionKind::Comment => {}
                ActionKind::Else => {
                    if let Some(Context::Branch { in_else, .. }) = stack.last_mut() {
                        *in_else = true;
                    }
                }
                ActionKind::End => {
                    if stack.len() <= 1 {
                        continue;
                    }
                    let ctx = stack.pop().unwrap();
                    if let Context::Branch {
                        kind,
                        pipe,
                        main,
                        else_nodes,
                        ..
                    } = ctx
                    {
                        let node = node_from_branch(kind, pipe, main, else_nodes);
                        current_nodes_mut(&mut stack).push(node);
                    }
                }
                ActionKind::RangeStart { pipe } => stack.push(Context::Branch {
                    kind: BranchKind::Range,
                    pipe,
                    main: Vec::new(),
                    else_nodes: Vec::new(),
                    in_else: false,
                }),
                ActionKind::IfStart { pipe } => stack.push(Context::Branch {
                    kind: BranchKind::If,
                    pipe,
                    main: Vec::new(),
                    else_nodes: Vec::new(),
                    in_else: false,
                }),
                ActionKind::WithStart { pipe } => stack.push(Context::Branch {
                    kind: BranchKind::With,
                    pipe,
                    main: Vec::new(),
                    else_nodes: Vec::new(),
                    in_else: false,
                }),
                ActionKind::Other => {
                    current_nodes_mut(&mut stack)
                        .push(Node::Action(ActionNode { raw: action.raw }));
                }
            },
        }
    }

    while stack.len() > 1 {
        let ctx = stack.pop().unwrap();
        if let Context::Branch {
            kind,
            pipe,
            main,
            else_nodes,
            ..
        } = ctx
        {
            let node = node_from_branch(kind, pipe, main, else_nodes);
            current_nodes_mut(&mut stack).push(node);
        }
    }

    match stack.pop().unwrap() {
        Context::List { nodes } => ListNode { nodes },
        Context::Branch { .. } => unreachable!(),
    }
}

fn node_from_branch(
    kind: BranchKind,
    pipe: String,
    main: Vec<Node>,
    else_nodes: Vec<Node>,
) -> Node {
    let branch = BranchNode {
        pipe,
        list: ListNode { nodes: main },
        else_list: if else_nodes.is_empty() {
            None
        } else {
            Some(ListNode { nodes: else_nodes })
        },
    };

    match kind {
        BranchKind::Range => Node::Range(branch),
        BranchKind::If => Node::If(branch),
        BranchKind::With => Node::With(branch),
    }
}

fn current_nodes_mut(stack: &mut Vec<Context>) -> &mut Vec<Node> {
    match stack.last_mut().expect("empty context stack") {
        Context::List { nodes } => nodes,
        Context::Branch {
            in_else,
            main,
            else_nodes,
            ..
        } => {
            if *in_else {
                else_nodes
            } else {
                main
            }
        }
    }
}

fn tokenize(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut idx = 0;
    while idx < input.len() {
        if let Some(open_rel) = input[idx..].find("{{") {
            let start = idx + open_rel;
            if start > idx {
                tokens.push(Token::Text(input[idx..start].to_string()));
            }

            let after_open = start + 2;
            if let Some(close_rel) = input[after_open..].find("}}") {
                let end = after_open + close_rel;
                let mut inner = &input[after_open..end];
                if inner.starts_with('-') {
                    inner = &inner[1..];
                }
                if inner.ends_with('-') {
                    inner = &inner[..inner.len() - 1];
                }
                let trimmed = inner.trim();
                let action_tokens = action_tokens(trimmed);
                for act in action_tokens {
                    tokens.push(Token::Action(act));
                }
                idx = end + 2;
            } else {
                tokens.push(Token::Text(input[start..].to_string()));
                break;
            }
        } else {
            tokens.push(Token::Text(input[idx..].to_string()));
            break;
        }
    }
    tokens
}

fn action_tokens(trimmed: &str) -> Vec<ActionToken> {
    let mut actions = Vec::new();
    if trimmed.is_empty() {
        return actions;
    }
    if trimmed.starts_with("/*") {
        actions.push(ActionToken {
            raw: String::new(),
            kind: ActionKind::Comment,
        });
        return actions;
    }
    if trimmed.starts_with("end") {
        actions.push(ActionToken {
            raw: trimmed.to_string(),
            kind: ActionKind::End,
        });
        return actions;
    }
    if trimmed.starts_with("else if") {
        actions.push(ActionToken {
            raw: "else".to_string(),
            kind: ActionKind::Else,
        });
        let rest = trimmed["else if".len()..].trim_start();
        if !rest.is_empty() {
            actions.push(ActionToken {
                raw: format!("if {}", rest),
                kind: ActionKind::Other,
            });
        }
        return actions;
    }
    if trimmed.starts_with("else") {
        actions.push(ActionToken {
            raw: trimmed.to_string(),
            kind: ActionKind::Else,
        });
        return actions;
    }
    if trimmed.starts_with("range") {
        let pipe = trimmed["range".len()..].trim_start().to_string();
        actions.push(ActionToken {
            raw: trimmed.to_string(),
            kind: ActionKind::RangeStart { pipe },
        });
        return actions;
    }
    if trimmed.starts_with("if") {
        let pipe = trimmed["if".len()..].trim_start().to_string();
        actions.push(ActionToken {
            raw: trimmed.to_string(),
            kind: ActionKind::IfStart { pipe },
        });
        return actions;
    }
    if trimmed.starts_with("with") {
        let pipe = trimmed["with".len()..].trim_start().to_string();
        actions.push(ActionToken {
            raw: trimmed.to_string(),
            kind: ActionKind::WithStart { pipe },
        });
        return actions;
    }

    actions.push(ActionToken {
        raw: trimmed.to_string(),
        kind: ActionKind::Other,
    });
    actions
}

#[derive(Debug)]
enum Ancestor<'a> {
    List(&'a ListNode),
    Range(&'a BranchNode),
}

fn traverse_list<'a>(
    list: &'a ListNode,
    ancestors: &mut Vec<Ancestor<'a>>,
    result: &mut Option<(String, String)>,
) {
    if result.is_some() {
        return;
    }
    ancestors.push(Ancestor::List(list));
    for node in &list.nodes {
        if result.is_some() {
            break;
        }
        match node {
            Node::Text(_) => {}
            Node::Action(action) => inspect_fields(&action.raw, ancestors, result),
            Node::Range(branch) => {
                ancestors.push(Ancestor::Range(branch));
                inspect_fields(&branch.pipe, ancestors, result);
                traverse_list(&branch.list, ancestors, result);
                if result.is_none() {
                    if let Some(else_list) = &branch.else_list {
                        traverse_list(else_list, ancestors, result);
                    }
                }
                ancestors.pop();
            }
            Node::If(branch) => {
                inspect_fields(&branch.pipe, ancestors, result);
                traverse_list(&branch.list, ancestors, result);
                if result.is_none() {
                    if let Some(else_list) = &branch.else_list {
                        traverse_list(else_list, ancestors, result);
                    }
                }
            }
            Node::With(branch) => {
                inspect_fields(&branch.pipe, ancestors, result);
                traverse_list(&branch.list, ancestors, result);
                if result.is_none() {
                    if let Some(else_list) = &branch.else_list {
                        traverse_list(else_list, ancestors, result);
                    }
                }
            }
        }
    }
    ancestors.pop();
}

fn inspect_fields<'a>(
    source: &str,
    ancestors: &mut Vec<Ancestor<'a>>,
    result: &mut Option<(String, String)>,
) {
    if result.is_some() {
        return;
    }
    let allow_field = source.trim_start().starts_with('.');
    for field in extract_fields(source) {
        if let Some(name) = field.first() {
            if name == "Thinking" && allow_field {
                if let Some(tags) = infer_from_ancestors(ancestors) {
                    *result = Some(tags);
                    return;
                }
            }
        }
    }
}

fn infer_from_ancestors<'a>(ancestors: &[Ancestor<'a>]) -> Option<(String, String)> {
    let mut nearest_range = None;
    for anc in ancestors.iter().rev() {
        if let Ancestor::Range(range) = anc {
            nearest_range = Some(*range);
            break;
        }
    }

    let range = nearest_range?;
    if !range_uses_field(&range, "Messages") {
        return None;
    }

    for anc in ancestors.iter().rev() {
        if let Ancestor::List(list) = anc {
            let opening = list.nodes.first().and_then(|n| match n {
                Node::Text(text) => Some(text.trim().to_string()),
                _ => None,
            });
            let closing = list.nodes.last().and_then(|n| match n {
                Node::Text(text) => Some(text.trim().to_string()),
                _ => None,
            });
            if opening.is_some() || closing.is_some() {
                return Some((opening.unwrap_or_default(), closing.unwrap_or_default()));
            }
        }
    }

    None
}

fn range_uses_field(range: &BranchNode, field: &str) -> bool {
    extract_fields(&range.pipe)
        .iter()
        .any(|idents| idents.first().map(|n| n == field).unwrap_or(false))
}

fn extract_fields(source: &str) -> Vec<Vec<String>> {
    let mut fields = Vec::new();
    let bytes = source.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'.' {
            i += 1;
        }
        if bytes[i] == b'.' {
            i += 1;
            if i >= bytes.len() {
                break;
            }
            if !(bytes[i] == b'_' || bytes[i].is_ascii_alphabetic()) {
                continue;
            }
            let mut idents = Vec::new();
            loop {
                let start = i;
                while i < bytes.len() && (bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric()) {
                    i += 1;
                }
                if start == i {
                    break;
                }
                idents.push(source[start..i].to_string());
                if i < bytes.len() && bytes[i] == b'.' {
                    i += 1;
                    if i >= bytes.len() {
                        break;
                    }
                    if !(bytes[i] == b'_' || bytes[i].is_ascii_alphabetic()) {
                        break;
                    }
                    continue;
                }
                break;
            }
            if !idents.is_empty() {
                fields.push(idents);
            }
            continue;
        }
        i += 1;
    }
    fields
}
