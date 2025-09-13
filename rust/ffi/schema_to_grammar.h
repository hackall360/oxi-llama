#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
int schema_to_grammar(const char *json_schema, char *grammar, size_t max_len);
#ifdef __cplusplus
}
#endif
