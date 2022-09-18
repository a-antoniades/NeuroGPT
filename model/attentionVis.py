import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class AttentionVis:
        '''attention Visualizer'''
        
        # def getAttention(self, spikes, n_Blocks):
        #         spikes = spikes.unsqueeze(0)
        #         b, t = spikes.size()
        #         token_embeddings = self.model.tok_emb(spikes)
        #         position_embeddings = self.model.pos_emb(spikes)
        #         # position_embeddings = self.model.pos_emb(spikes)
        #         x = token_embeddings + position_embeddings

        #         # aggregate attention from n_Blocks
        #         atts = None
        #         for n in n_Blocks:
        #                 attBlock = self.model.blocks[n].attn
        #                 attBlock(x).detach().numpy()    # forward model
        #                 att = attBlock.att.detach().numpy()
        #                 att = att[:, 1, :, :,].squeeze(0)
        #                 atts = att if atts is None else np.add(atts, att)
                
        #         # normalize
        #         atts = atts/len(n_Blocks)
        #         return atts
        
        def visAttention(att):
                plt.matshow(att)
                att_range = att.max()
                cb = plt.colorbar()
                cb.ax.tick_params()
                plt.show()
        
        
        @torch.no_grad()
        def getAttention(x, model, blocks=None):
                # take over whatever gpus are on the system
                device = 'cpu'
                model = model.eval()
                model.to(device)
                x = x.to(device)
                idx = x[:, 0].unsqueeze(0).long()
                dtx = x[:, 1].unsqueeze(0)
                b, t = idx.size()
                token_embeddings = model.tok_emb(idx).to(device)
                position_embeddings = model.pos_emb(idx).to(device) if model.config.pos_emb else 0
                temporal_embeddings = model.temp_emb(dtx).to(device) if model.config.temp_emb else 0
                # position_embeddings = self.model.pos_emb(spikes)
                x = token_embeddings + position_embeddings + temporal_embeddings

                # aggregate attention from n_Blocks
                atts = None
                n_blocks = model.config.n_layer
                blocks = range(n_blocks) if blocks is None else blocks
                for n in range(n_blocks):
                        attBlock = model.blocks[n].attn
                        attBlock(x).detach().numpy()    # forward model
                        att = attBlock.att.detach().numpy()
                        att = att[:, 1, :, :,].squeeze(0)
                        atts = att if atts is None else np.add(atts, att)
                
                # normalize
                atts = atts / n_blocks
                return atts


        def att_models(models, dataset, neurons):
                ''' 
                Input list of models
                Returns Attentions over dataset
                '''
                models_atts = []
                for model in models:
                        attention_scores = np.zeros(len(neurons))
                        data = dataset
                        pbar = tqdm(enumerate(data), total=len(data))
                        for it, (x, y) in pbar:
                                # scores = np.array(np.zeros(len(neurons)))
                                att = np.zeros(len(neurons))
                                score = AttentionVis.getAttention(x, model)
                                if score.size >= 1: score = score[-1]
                                # scores.append(score)
                                for idx, neuron in enumerate(x[:, 0]):
                                        """ 
                                        for each neuron in scores,
                                        add its score to the array
                                        """
                                        neuron = int(neuron.item())
                                        att[neuron] += score[idx]
                                attention_scores = np.vstack((attention_scores, att))
                                if it > len(dataset):
                                        models_atts.append(attention_scores.sum(axis=0))
                                        break
                return models_atts