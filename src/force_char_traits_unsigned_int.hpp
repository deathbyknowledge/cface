#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <climits>
#include <streambuf>
#include <string>

// emsdk's libc++ strictly adheres to the standard and doesn't
// provide an implementation of chair_traits<unsigned int>, which
// dlib needs. So I just provide one. I am aware this is UB.
namespace std
{
  template <>

  struct char_traits<unsigned int>
  {
    using char_type = unsigned int;
    using int_type = std::uint_least32_t;

    static void
    assign(char_type &c1, const char_type &c2) noexcept
    {
      c1 = c2;
    }

    static bool
    eq(const char_type &c1, const char_type &c2) noexcept
    {
      return c1 == c2;
    }

    static bool
    lt(const char_type &c1, const char_type &c2) noexcept
    {
      return c1 < c2;
    }

    static int
    compare(const char_type *s1, const char_type *s2, size_t n)
    {
      for (; n > 0; --n, ++s1, ++s2)
      {
        if (*s1 < *s2)
          return -1;
        if (*s2 < *s1)
          return 1;
      }
      return 0;
    }

    static size_t
    length(const char_type *s)
    {
      size_t i = 0;
      while (s[i] != 0)
      {
        ++i;
      }
      return i;
    }

    static const char_type *
    find(const char_type *s, size_t n, const char_type &a)
    {
      for (; n; --n)
      {
        if (*s == a)
        {
          return s;
        }
        ++s;
      }
      return nullptr;
    }

    static char_type *
    move(char_type *s1, const char_type *s2, size_t n)
    {
      // Overlapping moves are allowed
      return static_cast<char_type *>(memmove(s1, s2, n * sizeof(char_type)));
    }

    static char_type *
    copy(char_type *s1, const char_type *s2, size_t n)
    {
      // Overlapping is undefined; if you need overlap, use move
      return static_cast<char_type *>(memcpy(s1, s2, n * sizeof(char_type)));
    }

    static char_type *
    assign(char_type *s, size_t n, char_type a)
    {
      for (size_t i = 0; i < n; ++i)
      {
        s[i] = a;
      }
      return s;
    }

    static constexpr char_type
    to_char_type(const int_type &c) noexcept
    {
      return static_cast<char_type>(c);
    }

    static constexpr int_type
    to_int_type(const char_type &c) noexcept
    {
      return static_cast<int_type>(c);
    }

    static constexpr bool
    eq_int_type(const int_type &c1, const int_type &c2) noexcept
    {
      return c1 == c2;
    }

    static constexpr int_type
    eof() noexcept
    {
      return static_cast<int_type>(WEOF);
    }

    static constexpr int_type
    not_eof(const int_type &c) noexcept
    {
      return eq_int_type(c, eof()) ? 0 : c;
    }
  };
} // namespace std
