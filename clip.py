import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
# Contrastive Language-Image Pre-training (CLIP) 
class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        """token_embedding: Creates an embedding layer for tokens, 
        where n_vocab is the number of tokens in the vocabulary,
          and n_embd is the dimensionality of the token embeddings."""
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # A learnable weight matrix encodes the position information for each token
        """nn.Parameter: This makes the tensor a learnable parameter, meaning its values will be adjusted during training."""
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim) 
        x = self.token_embedding(tokens)
        # (Batch_Size, Seq_Len,Dim) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding
        
        return x

class CLIPLayer(nn.Module):
    # n_head (number of attention heads) and n_embd (embedding dimension).
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Self attention
        self.attention = SelfAttention(n_head, n_embd)
        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # Feedforward layer
        # experimentally found 4*n_emb
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # (Batch_Size, Seq_Len, Dim)
        residue = x
        
        ### SELF ATTENTION ###

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension. 

        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        x = self.linear_1(x)
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        #  GELU activation (x * torch.sigmoid(1.702 * x)
        x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        
        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)
        
        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        # in research work
        # embedding: Instantiates CLIPEmbedding with specific vocabulary size (49408), embedding dimension (768), and token count (77).
        self.embedding = CLIPEmbedding(49408, 768, 77)
 # """layers: Creates a list of CLIPLayer instances (nn.ModuleList) with 12 layers, each having 12 attention heads and 768 embedding dimensions."""
# nn.ModuleList, which is a PyTorch container for holding and managing a list of nn.Module instances.
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
            ])
        


        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)
        
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply encoder layers similar to the Transformer's encoder.
        for layer in self.layers: 
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)
        
        return output



"""Overall Functionality
Purpose: The CLIP model processes tokenized text inputs (tokens) through a series of transformer-like layers (CLIPLayer), which include self-attention and feedforward networks.
Architecture: It leverages embeddings (CLIPEmbedding), self-attention (SelfAttention), and layer normalization (LayerNorm) to encode token sequences into fixed-size embeddings (n_embd dimensions).
Output: Produces normalized embeddings (output) suitable for downstream tasks like image-text retrieval or classification."""