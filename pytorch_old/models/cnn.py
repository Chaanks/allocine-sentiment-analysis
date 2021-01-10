import torch


class CNN(torch.nn.Module):
    def __init__(
        self,
        num_words,
        embeddings_dim,
        padding_idx,
        num_filters,
        filters_dim, 
        out_dim, 
        dropout, 
    ):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(
            num_words,
            embeddings_dim, 
            padding_idx = padding_idx
        )
        
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(
                in_channels  = 1, 
                out_channels = num_filters, 
                kernel_size  = (dim, embeddings_dim)
            ) for dim in filters_dim
        ])
        
        self.fc = torch.nn.Linear(len(filters_dim) * num_filters, out_dim)
        
        self.dropout = torch.nn.Dropout(dropout)

            
    def forward(self, text):
        #text = [batch size, sent len]
        
        embedded = self.embedding(text) 
        # embedded = [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
            
        conved = [ torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs ]
        # conved_n = [batch size, num_filters, sent len - filters_dim[n] + 1] 
        pooled = [ torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved ]
        # pooled_n = [batch size, num_filters]
            
        cat = self.dropout(torch.cat(pooled, dim = 1))
        # cat = [batch size, num_filters * len(filters_dim)]

        return self.fc(cat)