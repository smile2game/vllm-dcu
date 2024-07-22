#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define OPT_SWITCH(COND, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static int opt = 1;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int opt = 2; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define NUM_THREADS_SWITCH(NUM_THREAD, ...)    \
  [&] {                                         \
    if (NUM_THREAD == 256) {                   \
      constexpr static int NUM_THREADS = 256;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int NUM_THREADS = 128;  \
      return __VA_ARGS__();                     \
    }                                           \
  }()

  // #define HEADSIZE_SWITCH(HEADDIM, ...)   \
  // [&] {                                    \
  //   if (HEADDIM == 64) {                   \
  //     constexpr static int HEAD_SIZE = 64;  \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM == 80) {            \
  //     constexpr static int HEAD_SIZE = 80;  \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM == 96) {            \
  //     constexpr static int HEAD_SIZE = 96;  \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM == 112) {           \
  //     constexpr static int HEAD_SIZE = 112; \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM == 128) {           \
  //     constexpr static int HEAD_SIZE = 128; \
  //     return __VA_ARGS__();                \
  //   } else if (HEADDIM == 256) {           \
  //     constexpr static int HEAD_SIZE = 256; \
  //     return __VA_ARGS__();                \
  //   }                                      \
  //   else {                                 \
  //     TORCH_CHECK(false, "Unsupported head size: ", HEADDIM);\
  //   }                                      \
  // }()

  #define HEADSIZE_SWITCH(HEADDIM, ...)   \
  [&] {                                    \
    if (HEADDIM == 128) {           \
      constexpr static int HEAD_SIZE = 128; \
      return __VA_ARGS__();                \
    } else {                                 \
      TORCH_CHECK(false, "Unsupported head size: ", HEADDIM);\
    }                                      \
  }()

#define REUSEKV_SWITCH(num_blocks , ...)      \
[&] {                                                   \
    if (num_heads % 2 == 0 && num_heads / num_kv_heads >= 4 && num_blocks >= 1200){      \
        constexpr static int REUSE_KV_TIMES = 4;        \
        return __VA_ARGS__();                           \
    } else if (num_heads / num_kv_heads >= 2 && num_blocks >= 1200){\
        constexpr static int REUSE_KV_TIMES = 2;        \
        return __VA_ARGS__();                           \
    } else {                                            \
        constexpr static int REUSE_KV_TIMES = 1;        \
        return __VA_ARGS__();                           \
    }                                                   \
}()

#define REUSEKV_SWITCH_V1(num_blocks , ...)      \
[&] {                                                   \
    if (num_heads > num_kv_heads && num_blocks >= 1200){      \
        constexpr static int REUSE_KV_TIMES = 2;        \
        return __VA_ARGS__();                           \
    }  else {                                           \
        constexpr static int REUSE_KV_TIMES = 1;        \
        return __VA_ARGS__();                           \
    }                                                   \
}()

