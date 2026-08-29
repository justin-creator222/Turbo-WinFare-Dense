#ifndef G4DENSE_C_API_H
#define G4DENSE_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef G4DENSE_BUILD_DLL
    #define G4DENSE_API __declspec(dllexport)
  #else
    #define G4DENSE_API __declspec(dllimport)
  #endif
#else
  #define G4DENSE_API __attribute__((visibility("default")))
#endif

G4DENSE_API void* g4dense_engine_create(const char* container_path);
G4DENSE_API void g4dense_engine_destroy(void* handle);

G4DENSE_API const char* g4dense_engine_generate(void* handle, const char* prompt, int max_tokens);
G4DENSE_API const char* g4dense_engine_get_telemetry(void* handle);

typedef struct {
    int max_tokens;
    float temperature;
    float top_p;
    int top_k;
    float repetition_penalty;
    int has_seed;
    unsigned long long seed;
    const char* system_prompt;
    const char* const* stop_strings;
    int stop_count;
    int tier_id;
} G4DenseGenerationOptions;

G4DENSE_API void g4dense_options_default(G4DenseGenerationOptions* out);

typedef int (*G4DenseTokenCallback)(void* user, int index, unsigned int token_id, const char* delta);

G4DENSE_API const char* g4dense_engine_generate_ex(void* handle, const char* messages_json,
                                                   const G4DenseGenerationOptions* opts,
                                                   G4DenseTokenCallback on_token, void* user);

G4DENSE_API int g4dense_engine_last_stop_reason(void* handle);
G4DENSE_API const char* g4dense_engine_last_error(void);
G4DENSE_API int g4dense_engine_switch_tier(void* handle, int tier_id);
G4DENSE_API void g4dense_engine_stop(void* handle);

#ifdef __cplusplus
}
#endif

#endif // G4DENSE_C_API_H
