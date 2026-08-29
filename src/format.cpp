#include <g4dense/format.hpp>
#include <sstream>

namespace g4dense {

void validate_header(const G4DenseHeader& header, uint64_t file_size) {
    if (header.magic != G4DenseHeader::EXPECTED_MAGIC) {
        std::stringstream ss;
        ss << "Invalid magic: expected 0x" << std::hex << G4DenseHeader::EXPECTED_MAGIC
           << ", got 0x" << header.magic;
        throw G4DenseFormatError(ss.str());
    }

    if (header.version != G4DenseHeader::EXPECTED_VERSION) {
        std::stringstream ss;
        ss << "Unsupported container version: expected " << G4DenseHeader::EXPECTED_VERSION
           << ", got " << header.version;
        throw G4DenseFormatError(ss.str());
    }

    if (header.num_layers == 0 || header.num_layers > G4DenseHeader::MAX_LAYERS) {
        std::stringstream ss;
        ss << "Invalid layer count: " << header.num_layers
           << " (max " << G4DenseHeader::MAX_LAYERS << ")";
        throw G4DenseFormatError(ss.str());
    }

    if (header.d_model == 0 || header.d_ff == 0 || header.vocab_size == 0) {
        throw G4DenseFormatError("Invalid model geometry dimensions (d_model, d_ff, or vocab_size is zero)");
    }

    // Alignment verification
    if (header.embed_offset % G4DenseHeader::ALIGNMENT_BYTES != 0) {
        throw G4DenseFormatError("Embedding offset is not 4096-byte aligned");
    }

    if (header.lm_head_offset % G4DenseHeader::ALIGNMENT_BYTES != 0) {
        throw G4DenseFormatError("LM head offset is not 4096-byte aligned");
    }

    for (uint32_t i = 0; i < header.num_layers; ++i) {
        if (header.layer_offsets[i] % G4DenseHeader::ALIGNMENT_BYTES != 0) {
            std::stringstream ss;
            ss << "Layer " << i << " offset (" << header.layer_offsets[i] << ") is not 4096-byte aligned";
            throw G4DenseFormatError(ss.str());
        }

        if (header.layer_sizes[i] == 0) {
            std::stringstream ss;
            ss << "Layer " << i << " has size 0";
            throw G4DenseFormatError(ss.str());
        }

        checked_add(header.layer_offsets[i], header.layer_sizes[i], "layer bounds");

        if (file_size > 0) {
            if (header.layer_offsets[i] + header.layer_sizes[i] > file_size) {
                std::stringstream ss;
                ss << "Layer " << i << " extends beyond file bounds (offset=" << header.layer_offsets[i]
                   << ", size=" << header.layer_sizes[i] << ", file_size=" << file_size << ")";
                throw G4DenseFormatError(ss.str());
            }
        }
    }
}

} // namespace g4dense
