#ifndef DICTIONARY_H
#define DICTIONARY_H

#include <stdio.h>
#include "../ds/emhash_map.hpp"
#include <string>
#include <array>
#include <list>

namespace preprocessor {

class Dictionary {
 public:
  Dictionary(FILE* dictionary, bool encode, bool decode);
  void Encode(FILE* input, int len, FILE* output);
  void EncodeString(const std::string& input, std::string* output);
  unsigned char Decode(FILE* input);

 private:
  void EncodeWord(const std::string& word, int num_upper, bool next_lower,
      FILE* output);
  void EncodeWordString(const std::string& word, int num_upper,
      bool next_lower, std::string* output);
  bool EncodeSubstring(const std::string& word, FILE* output);
  bool EncodeSubstringString(const std::string& word, std::string* output);
  void AddToBuffer(FILE* input);

  emhash6::HashMap<std::string, unsigned int> byte_map_;
  emhash6::HashMap<unsigned int, std::string> reverse_map_;
  std::list<unsigned char> output_buffer_;
  bool decode_upper_ = false, decode_capital_ = false;
  unsigned int longest_word_ = 0;
};

}

#endif
