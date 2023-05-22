#include "utils.h"
#include "json.h"
#include "tokenizer/bpe.h"
#include "tokenizer/mpt_tokenizer_config.h"
#include "tokenizer/gptj_tokenizer_config.h"

#include <fstream>
#include <regex>
#include <stdexcept>

void get_bpecpp_tokenizer(const TokenizerType ttype, std::unique_ptr<bpecpp::BPE>& bpe, std::unique_ptr<bpecpp::AdditionalVocabAdapter>& av) {
    std::vector<bpecpp::additional_vocab_item> avis;
    std::unordered_map<std::string, uint32_t> vocab;
    std::vector<std::string> merges;

    uint32_t tok_id = 0;
    switch (ttype) {
        case TokenizerType::MPT_CHAT:
            avis.push_back({ .id = 50277, .content = "<|im_start|>", .special = true });
            avis.push_back({ .id = 50278, .content = "<|im_end|>", .special = true });
        case TokenizerType::MPT:
            avis.insert(avis.end(), mpt_additional_vocab.begin(), mpt_additional_vocab.end());
            for (const char* cchar: mpt_vocab) {
                vocab.insert({std::string(cchar, std::strlen(cchar)), tok_id++ });
            }
            for (const char* cchar: mpt_merges) {
                merges.push_back(std::string(cchar, std::strlen(cchar))); 
            }
        break;
        case TokenizerType::GPTJ:
            avis.insert(avis.end(), gptj_additional_vocab.begin(), gptj_additional_vocab.end());
            for (const char* cchar: gptj_vocab) {
                vocab.insert({std::string(cchar, std::strlen(cchar)), tok_id++ }); 
            }
            for (const char* cchar: gptj_merges) {
                merges.push_back(std::string(cchar, std::strlen(cchar)));
            }
        break;
        default:
            throw std::invalid_argument("invalid tokenizer type");
    }
    av = std::make_unique<bpecpp::AdditionalVocabAdapter>(avis);
    bpe = std::make_unique<bpecpp::BPE>(vocab, merges);
}

gpt_vocab::id gpt_sample_top_k_top_p(
        const gpt_vocab & vocab,
        const size_t actualVocabSize,
        const int32_t * last_n_tokens_data,
        int   last_n_tokens_size,
        const std::vector<float> logits,
        int    top_k,
        double top_p,
        double temp,
        float repeat_penalty,
        std::mt19937 & rng) {
    int n_logits = actualVocabSize;

    const auto last_n_tokens = std::vector<int32_t>(last_n_tokens_data, last_n_tokens_data + last_n_tokens_size);
    const auto * plogits = logits.data() + logits.size() - n_logits;

    std::vector<std::pair<double, gpt_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const float scale = 1.0f/temp;
        for (int i = 0; i < n_logits; ++i) {
            // repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
            // credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
            if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
                // if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if (plogits[i] < 0.0f) {
                    logits_id.push_back(std::make_pair(plogits[i]*scale*repeat_penalty, i));
                } else {
                    logits_id.push_back(std::make_pair(plogits[i]*scale/repeat_penalty, i));
                }
            } else {
                logits_id.push_back(std::make_pair(plogits[i]*scale, i));
            }
        }
    }

    // find the top K tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<double, gpt_vocab::id> & a, const std::pair<double, gpt_vocab::id> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);

    double maxl = -INFINITY;
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    std::vector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                top_k = i + 1;
                probs.resize(top_k);
                logits_id.resize(top_k);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //printf("\n");
    //for (int i = 0; i < (int) probs.size(); i++) {
    //    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
    //}
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}
