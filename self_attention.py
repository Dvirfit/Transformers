import numpy as np
import matplotlib.pyplot as plt
import time

def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def self_attention(embeddings):
    """
    Compute single-head self-attention with self-connection bias.
    Args:
        query: Input query matrix (batch_size, seq_len, d_k)
        key: Input key matrix (batch_size, seq_len, d_k)
        value: Input value matrix (batch_size, seq_len, d_k)
    Returns:
        output: Attention output
        attention_weights: Attention weight matrix
    """
    W_value, W_key, W_query, Attention_Weights, Output_Data, x = convert_q510_matrices_to_float()

    # Create input matrices (batch_size, seq_len, d_model)
    W_query = np.random.rand(8, 24)
    W_key = np.random.rand(8, 24)
    W_value = np.random.rand(8, 24)

    print(f"W_query shape: {W_query.shape}, W_key shape: {W_key.shape}, W_value shape: {W_value.shape}")
    print(f"W_query: {W_query}\nW_key: {W_key}\nW_value: {W_value}\n")

    # embeddings = x
    key = np.matmul(embeddings, W_key)
    query = np.matmul(embeddings, W_query)
    value = np.matmul(embeddings, W_value)



    key = key.transpose()
    d_k = query.shape[-1]
    # print(f"query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
    scores = np.matmul(query, key) / np.sqrt(d_k)
    print(f"scores shape: {scores.shape}\n scores: {scores}")

    # Apply softmax
    #attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    attention_weights =softmax(scores)

    output = np.matmul(attention_weights, value)
    return output, attention_weights


def plot_attention_heatmap(attention_weights, words, title, filename):
    """
    Plot attention weights as a heatmap with word labels.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(attention_weights, cmap='viridis', interpolation='nearest')
    plt.xticks(np.arange(len(words)), words, rotation=45, ha='right')
    plt.yticks(np.arange(len(words)), words)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title(title)
    plt.colorbar(label='Attention Weight')
    plt.savefig(filename)
    plt.close()

def convert_matrix(matrix):
    fractional_scale = 1 / 1024
    float_matrix = []
    for row in matrix:
        float_row = []
        for hex_str in row:
            int_val = int(hex_str, 16)
            if int_val > 0x7FFF:
                int_val -= 0x10000
            float_val = int_val * fractional_scale
            float_row.append(float_val)
        float_matrix.append(float_row)
    return float_matrix

def convert_q510_matrices_to_float():
    """
    Converts each matrix (W_query, W_key, W_value, Attention Weights, Output Data, and x)
    from Q5.10 hexadecimal to floating-point numbers

    Returns:
        W_query, W_key, W_value, Attention Weights, Output Data, and x
    """
    # Define all matrices
    w_query = [
        ["ffff", "9972", "899d", "efed", "c134", "6c05", "6b7d", "35e5", "63bb", "52e8", "7dd0", "f449", "68fe", "2a5b",
         "414f", "ad78", "9b0d", "5938", "efa9", "d742", "d3f5", "2161", "4c88", "3c1c"],
        ["5513", "fd19", "6e40", "a7f0", "778c", "816a", "ed6a", "8229", "28d6", "12e2", "59c6", "1c0d", "24d4", "1741",
         "9b97", "b3ac", "b4a7", "5255", "66e7", "49bf", "b41d", "c02d", "dc22", "a01a"],
        ["2dc7", "e7fd", "c028", "1eb9", "b5f6", "2633", "042e", "f449", "fde1", "400b", "7af5", "0c64", "ef16", "0a0c",
         "8a17", "2729", "669b", "3d44", "a7b1", "f069", "b669", "fdf0", "2cc4", "322f"],
        ["370d", "6dfa", "a20f", "19ee", "bf6e", "4b96", "3041", "86c8", "6c1a", "2493", "ccda", "7a6f", "3d04", "fa4a",
         "8a4e", "31fb", "fb98", "d62d", "4a54", "57ce", "3519", "4223", "dc31", "b674"],
        ["3111", "d964", "b3f1", "bb9a", "57f5", "42f6", "130b", "c5dd", "3229", "1c0d", "ba56", "4241", "7f82", "2e48",
         "b984", "0f4e", "789d", "73a4", "3572", "e60f", "01f8", "5784", "01c2", "a93a"],
        ["368d", "45bd", "e540", "b602", "905a", "4139", "f12e", "a2ae", "272c", "17e3", "6d12", "6ede", "65d2", "cebe",
         "ea68", "cef1", "e7f1", "c48b", "de59", "ce54", "370f", "ef14", "1e27", "19de"],
        ["c2d7", "e4ed", "e7cc", "b9a7", "a48d", "087b", "58b6", "19df", "9a6a", "d905", "718a", "8a1d", "f61d", "433a",
         "1bd6", "5123", "80bc", "e3f2", "8a2d", "17f0", "db66", "9186", "f1aa", "99d1"],
        ["b6e0", "515c", "ecc8", "da5f", "ddd1", "622c", "d01d", "6ec0", "5025", "ded5", "aaea", "61d3", "84ad", "3550",
         "38f9", "6b43", "82d1", "cdb3", "2b42", "8ab5", "e7f3", "0923", "cb41", "2cc7"],
        ["2960", "275f", "624d", "cae5", "fd59", "1e85", "71de", "07c7", "121f", "b0f7", "6b60", "7605", "9c18", "8c36",
         "0788", "4cb4", "e57d", "754e", "9bc0", "e0fb", "6f55", "8fa4", "8847", "5972"]
    ]

    w_key = [
        ["3ffe", "2716", "d4dc", "bb8a", "3038", "ef33", "4902", "fab7", "739c", "c11d", "7cd3", "6d82", "705e", "9d02",
         "eb03", "8bf4", "2b95", "50bb", "22fc", "85b6", "dd7c", "36a2", "95ed", "18fb"],
        ["d59d", "d52d", "66a7", "cf25", "b752", "579a", "df3b", "2c80", "8a19", "4e9a", "05f2", "b67e", "fd22", "c314",
         "aa33", "6d06", "7f27", "5628", "b69e", "e15f", "fde6", "ba7d", "ca7a", "94a4"],
        ["115e", "a5bf", "9160", "2995", "58b1", "8ab1", "3e89", "051d", "e65a", "d110", "1089", "bd2d", "80d6", "98a0",
         "8d74", "d5ca", "8e67", "60cd", "50c4", "4e1e", "4e09", "17ce", "5560", "59f9"],
        ["1ede", "e7fa", "9901", "a686", "7c7b", "9d08", "f67a", "b347", "41b8", "43e5", "51c1", "7777", "0ebf", "d184",
         "234f", "dc6a", "ab9e", "c34f", "742f", "b193", "5631", "d147", "b7be", "110b"],
        ["6359", "3b62", "5443", "dee3", "127f", "3e8b", "f4f2", "e997", "fad3", "3621", "2939", "40d7", "bd22", "091f",
         "aeaa", "8dcc", "4cf8", "4ea5", "5553", "a36a", "deb9", "28a7", "8f63", "b0ca"],
        ["4eaa", "4f8c", "ea4e", "361e", "7d1b", "6ef6", "9ea9", "5b8d", "b0f4", "dd9d", "a897", "ef99", "ac1e", "84e9",
         "d2fb", "5316", "40da", "6994", "4377", "be2f", "285b", "1d70", "fd98", "6b04"],
        ["5d64", "c237", "8b85", "092a", "6b34", "d8c1", "d8b8", "0ddd", "c127", "eafa", "f9fd", "0b5b", "2dcf", "62c6",
         "de88", "5a1b", "5bad", "da88", "f92f", "b4b6", "1d3b", "994a", "5fd9", "f3d6"],
        ["48f1", "ae73", "b541", "d462", "2ed0", "f937", "b220", "7163", "8f0f", "fa5a", "1f13", "caae", "b00c", "860c",
         "6a0c", "2e36", "f042", "bc2e", "12f0", "858d", "c712", "4e8b", "48bd", "4739"],
        ["bbcf", "e3d3", "1027", "1832", "cd65", "d341", "5fe7", "d4cc", "7b10", "c748", "a5c7", "b468", "edac", "a6ea",
         "f1de", "07d4", "66c2", "c0e5", "de74", "7d86", "6586", "9699", "36e0", "2e7b"]
    ]

    w_value = [
        ["d66c", "cf68", "8c9c", "c6de", "5e1d", "a30a", "17bd", "3d18", "5d19", "0d34", "c763", "cbde", "ea97", "dff2",
         "5ed5", "2d7b", "7eef", "c80f", "3a73", "0134", "5c4b", "b769", "0256", "af1e"],
        ["62bb", "4f73", "ae58", "3af5", "64c7", "779a", "4fa5", "4af7", "d5c5", "a3f2", "e324", "5880", "5aaf", "36f6",
         "0383", "ba3a", "362f", "f0fe", "6836", "190a", "a311", "3ff0", "f117", "038e"],
        ["ead4", "c21d", "f141", "8aec", "1a87", "a794", "6574", "d52c", "a766", "ce1b", "9e12", "74a0", "8b0c", "5fe0",
         "2d50", "8c4e", "49ff", "b9aa", "6207", "eeb8", "d34d", "da62", "59c1", "8985"],
        ["07bf", "2c99", "3d64", "41c6", "17f0", "971a", "8066", "f78c", "b4f7", "d1e6", "0ce6", "2375", "c66b", "83d6",
         "d50c", "9d92", "d0ec", "82bd", "59e2", "d379", "7dab", "cd49", "c392", "d720"],
        ["6e6f", "6fea", "ba9a", "96bc", "3956", "c8eb", "f1c5", "dec7", "a926", "b512", "a840", "4733", "bc60", "ac6c",
         "7d60", "5902", "ed88", "e6da", "b34c", "6fe3", "dd7d", "f77d", "41f5", "a2b0"],
        ["af05", "a74f", "2ef7", "701c", "7fec", "e641", "37d8", "93cf", "8bbc", "05c0", "3537", "6bd7", "27d4", "715d",
         "7f3e", "2a49", "29c4", "c7f7", "e121", "b27d", "c056", "8617", "18d0", "f295"],
        ["22d1", "88c9", "0083", "b17e", "c27e", "29e7", "1b4d", "770d", "e845", "2457", "4dc1", "42a4", "4efc", "d5f9",
         "7d80", "b084", "252a", "2377", "a31a", "0ec4", "96c3", "1feb", "db27", "b9b3"],
        ["d6af", "2780", "7555", "11f8", "4c79", "f256", "ecf4", "9241", "9ce7", "b356", "52d7", "ca40", "47c2", "87d9",
         "1b94", "9537", "e65d", "c4cd", "3a81", "f626", "2537", "06e8", "a764", "338a"],
        ["a3a8", "a74b", "a01a", "b521", "5b35", "b202", "0430", "5491", "6c24", "9a39", "d0fe", "9dcf", "63f8", "e843",
         "69e7", "c5c8", "b82c", "4471", "f550", "8bfe", "8c06", "b4f4", "f35d", "f41f"]
    ]

    attention_weights = [
        ["00a8", "0066", "00f4", "00f4", "00f4", "0013"],
        ["005d", "00de", "0027", "00de", "00de", "00de"],
        ["00aa", "00aa", "00aa", "00aa", "00aa", "00aa"],
        ["00f6", "001f", "0005", "00f6", "00f6", "00f6"],
        ["001b", "00f3", "00f3", "00f3", "0015", "00f3"],
        ["00ed", "0013", "0143", "0010", "0068", "0143"]
    ]

    output_data = [
        ["e6e2", "2748", "7ba6", "c127", "264a", "f0a4", "30b2", "2ae8", "c3f4", "489a", "1c6c", "f0b0", "d993", "43a4",
         "3b85", "30b2", "1a54", "dedc", "21d2", "0ece", "dfb1", "c388", "7522", "997d"],
        ["00e0", "353b", "7ff3", "d5a6", "473c", "a736", "e184", "a7ad", "0555", "d028", "bd34", "e6a2", "9f39", "a1bc",
         "b19e", "f8e1", "662e", "e563", "5228", "9090", "f1ef", "9951", "3067", "fa38"],
        ["ed2e", "e674", "3e32", "00b4", "67d0", "7c58", "db2c", "1bda", "69c8", "72a4", "c318", "73e2", "de20", "62a4",
         "27ac", "449e", "15f4", "2564", "39f0", "a6a6", "a814", "9f40", "5bf2", "4310"],
        ["0769", "0d94", "2c34", "eba7", "1625", "ef6a", "2642", "9589", "24a8", "5136", "d118", "7db8", "683a", "b662",
         "c87e", "1dca", "d514", "2fc8", "8603", "424b", "0890", "7b9a", "953d", "708b"],
        ["123b", "7060", "e231", "83b0", "331c", "b564", "6b0c", "6ca5", "3bf2", "5cfe", "7300", "4255", "7340", "dd6a",
         "7872", "c1ab", "6924", "5a52", "69a6", "ab11", "93ae", "a068", "8f95", "b374"],
        ["b978", "2a3b", "1f91", "375e", "1030", "879e", "563d", "e066", "7f1d", "1d09", "4567", "d926", "7a3d", "7e66",
         "176d", "3b15", "9fd4", "b765", "9410", "0942", "6dd4", "7e17", "9ba7", "c1c7"]
    ]

    x = [
        ["FB98", "03FD", "0122", "F9CD", "FDAA", "06AE", "F65C", "FE4B", "0519"],
        ["FC87", "FD4D", "FF9E", "0609", "FD3C", "FE5F", "FE45", "08E2", "08D2"],
        ["0404", "018B", "02F5", "0608", "F83B", "04C0", "FAF7", "FD3C", "03A9"],
        ["FA48", "FFF0", "FC84", "FFE9", "F4C2", "F8D2", "FD4C", "03B7", "FF4D"],
        ["0003", "02C8", "FC76", "0121", "FCCB", "F8FA", "FEAD", "024E", "015B"],
        ["FFF4", "09B3", "01A8", "03F8", "08E7", "FACD", "FBC6", "070B", "FCE8"]
    ]



    # Convert all matrices and store in a dictionary
    W_query = convert_matrix(w_query)
    W_key = convert_matrix(w_key)
    W_value = convert_matrix(w_value)
    Attention_Weights = convert_matrix(attention_weights)
    Output_Data = convert_matrix(output_data)
    x =  convert_matrix(x)
    return W_value, W_key, W_query, Attention_Weights, Output_Data, x

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    batch_size = 1
    d_model = 8  # Embedding dimension
    sentence = "Life is short eat dessert first"
    words = sentence.split()
    seq_len = len(words)

    # Create fixed word embeddings (random for simplicity)
    embeddings = np.random.randn(seq_len, d_model)
    print(f"Input shape:{embeddings.shape}Input Matrix:{embeddings}\n")

    # start the performance counter
    start_time = time.perf_counter()

    # Compute self-attention
    output, attention_weights = self_attention(embeddings)

    # end the performance counter
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for self-attention computation: {elapsed_time:.6f} seconds")

    # Print output matrix
    print(f"\nOutput shape:{output.shape}Output Matrix:{output}")


    # Plot attention weights as heatmap
    plot_attention_heatmap(
        attention_weights,
        words,
        'Self-Attention Weights for Sentence',
        'sentence_attention_heatmap.png'
    )


if __name__ == "__main__":
    main()