import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    return model.encode(texts, normalize_embeddings=True)

def quantize_to_binary(embeddings):
    binary = (embeddings > 0).astype(np.uint8)
    packed = np.packbits(binary, axis=1)
    return packed, binary

def hamming_distance_raw(a, b):
    # raw bit difference count
    return np.unpackbits(a ^ b).sum()

def verify_quality(texts):
    print("generating float embeddings...")
    embeddings = get_embeddings(texts)
    
    print("quantizing to binary...")
    packed, binary = quantize_to_binary(embeddings)
    
    print("\n--- ranking check ---")
    
    correct = 0
    total = 0
    
    for i in range(len(texts)):
        float_sims = []
        ham_sims = []
        
        for j in range(len(texts)):
            if i == j:
                continue
            
            float_sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[j].reshape(1, -1)
            )[0][0]
            float_sims.append((j, float_sim))
            
            ham = hamming_distance_raw(packed[i], packed[j])
            ham_sims.append((j, -ham))  # negative so less distance = more similar
        
        float_ranked = [x[0] for x in sorted(float_sims, key=lambda x: -x[1])]
        ham_ranked = [x[0] for x in sorted(ham_sims, key=lambda x: -x[1])]
        
        top1_correct = float_ranked[0] == ham_ranked[0]
        if top1_correct:
            correct += 1
        total += 1
        
        print(f"\nquery: '{texts[i][:40]}'")
        print(f"  float top match:   '{texts[float_ranked[0]][:40]}'")
        print(f"  hamming top match: '{texts[ham_ranked[0]][:40]}'")
        print(f"  top 1: {'✓ correct' if top1_correct else '✗ wrong'}")
        
        # show full ranking comparison
        print(f"  float ranking:   {[texts[x][:15].strip() for x in float_ranked]}")
        print(f"  hamming ranking: {[texts[x][:15].strip() for x in ham_ranked]}")
    
    print(f"\n--- summary ---")
    print(f"top 1 accuracy: {correct}/{total} ({(correct/total)*100:.0f}%)")
    print(f"binary embeddings {'are' if correct/total > 0.7 else 'are NOT'} good enough to proceed")

if __name__ == "__main__":
    texts = [
        "the robot motor requires high torque",
        "motor needs strong rotational force",
        "the weather today is sunny and warm",
        "robot arm joint needs calibration",
        "it is a beautiful day outside"
    ]
    
    verify_quality(texts)