use parser::thinking::infer_tags;

#[test]
fn infer_basic() {
    let tmpl = "{{ if .Thinking}}/think{{ end }}{{- range $i, $_ := .Messages }}{{- $last := eq (len (slice $.Messages $i)) 1 -}}{{ if and $last .Thinking }}<think>{{ .Thinking }}</think>{{ end }}{{ end }}";
    let (o, c) = infer_tags(tmpl);
    assert_eq!(o, "<think>");
    assert_eq!(c, "</think>");
}

#[test]
fn infer_whitespace_trimmed() {
    let tmpl = "{{ if .Thinking}}/think{{ end }}{{- range $i, $_ := .Messages }}{{- $last := eq (len (slice $.Messages $i)) 1 -}}{{ if and $last .Thinking }}Some text before   {{ .Thinking }}    Some text after{{ end }}{{ end }}";
    let (o, c) = infer_tags(tmpl);
    assert_eq!(o, "Some text before");
    assert_eq!(c, "Some text after");
}

#[test]
fn infer_qwen3() {
    let tmpl = "{{- if or .System .Tools .Thinking }}<|im_start|>system{{- if .System }}{{ .System }}{{- end }}{{- if .Tools }}<tools>{{- range .Tools }}{{ .Function }}{{- end }}</tools>{{- end }}{{- if .Thinking }}/think{{- else }}/no_think{{- end }}<|im_end|>{{ end }}{{- range $i, $_ := .Messages }}{{- $last := eq (len (slice $.Messages $i)) 1 -}}{{- if eq .Role \"user\" }}<|im_start|>user{{ .Content }}<|im_end|>{{ else if eq .Role \"assistant\" }}<|im_start|>assistant{{ if and $last .Thinking }}<think>{{ .Thinking }}</think>{{ end }}{{ if .Content }}{{ .Content }}{{ end }}{{- end }}{{- end }}";
    let (o, c) = infer_tags(tmpl);
    assert_eq!(o, "<think>");
    assert_eq!(c, "</think>");
}
#[test]
fn infer_doubly_nested_range() {
    let tmpl = "{{ if .Thinking}}/think{{ end }}{{- range $i, $_ := .Messages }}{{- range $j, $_ := .NotMessages }}{{- $last := eq (len (slice $.Messages $i)) 1 -}}{{ if and $last .Thinking }}<think>{{ .Thinking }}</think>{{ end }}{{ end }}{{ end }}";
    let (o, c) = infer_tags(tmpl);
    assert!(o.is_empty() && c.is_empty());
}
