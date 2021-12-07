import copy

import numpy as np
import plotly.graph_objects as go
import scipy

from utils import fill_text_blank


class InfGraph():

    def __init__(self, examples, num_word, num_layer, mask_index, inf,
                 comb_greedy_val, comb_greedy_attn, heatmap, attn_heatmap,
                 indices_attn):
        self.examples = examples
        self.num_word = num_word
        self.num_layer = num_layer
        self.mask_index = mask_index
        self.inf = inf
        self.comb_greedy_val = comb_greedy_val
        self.comb_greedy_attn = comb_greedy_attn
        self.heatmap = heatmap
        self.attn_heatmap = attn_heatmap
        self.indices_attn = indices_attn

    def plot_distribution(self, ind):
        eind = ind[0]
        traces = [
            go.Bar(y=fill_text_blank(self.examples[eind].tokens, self.num_word),
                   x=self.inf[ind].mean(0),
                   name='input influence',
                   orientation='h'),
            go.Bar(y=fill_text_blank(self.examples[eind].tokens, self.num_word),
                   x=self.comb_greedy_val[ind].mean(0),
                   name='word-level influence',
                   orientation='h'),
            go.Bar(y=fill_text_blank(self.examples[eind].tokens, self.num_word),
                   x=self.comb_greedy_attn[ind].mean(0),
                   name='attention-head path influence',
                   orientation='h')
        ]
        return traces

    def hm2layer_connection(self,
                            ind,
                            scaler=3,
                            plot_attention_connection=True):

        def mode(lst):
            if len(lst) == 0:
                return -2, 0
            else:
                return scipy.stats.mode(lst)[0][0], scipy.stats.mode(lst)[1][0]

        layer_connection_pos = np.zeros(
            (self.num_layer, self.num_word, self.num_word))
        layer_connection_neg = np.zeros(
            (self.num_layer, self.num_word, self.num_word))
        attention_connection = [[[[]
                                  for _ in range(self.num_word)]
                                 for _ in range(self.num_word)]
                                for _ in range(self.num_layer)]
        heatmap = copy.deepcopy(self.heatmap)
        if self.num_layer == 6:
            heatmap[:, self.mask_index, :, :] = 0

        for i in ind:
            tmp_idx = np.where(heatmap[i] == 1)
            idx3 = tmp_idx[2].reshape(-1, self.num_layer)
            idx4 = np.hstack((tmp_idx[0].reshape(-1,
                                                 self.num_layer)[:, 0][:, None],
                              idx3[:, :-1])).flatten()
            for z1, z2, z3, z4 in zip(tmp_idx[1], idx4, tmp_idx[2], tmp_idx[0]):
                layer_connection_pos[z1, z2, z3] += 1
        for i in ind:
            tmp_idx = np.where(heatmap[i] == -1)
            idx3 = tmp_idx[2].reshape(-1, self.num_layer)
            idx4 = np.hstack((tmp_idx[0].reshape(-1,
                                                 self.num_layer)[:, 0][:, None],
                              idx3[:, :-1])).flatten()
            for z1, z2, z3, z4 in zip(tmp_idx[1], idx4, tmp_idx[2], tmp_idx[0]):

                layer_connection_neg[z1, z2, z3] += -1
        if plot_attention_connection:
            for i in ind:
                try:
                    tmp_idx = np.where(self.attn_heatmap[i] >= -1)
                    idx3 = tmp_idx[2].reshape(-1, self.num_layer)
                    idx4 = np.hstack(
                        (tmp_idx[0].reshape(-1, self.num_layer)[:, 0][:, None],
                         idx3[:, :-1])).flatten()
                    for z1, z2, z3, z4 in zip(tmp_idx[1], idx4, tmp_idx[2],
                                              tmp_idx[0]):
                        attention_connection[z1][z2][z3].append(
                            int(self.indices_attn[i][z1][z4]))
                except:
                    pass
            attention_connection = np.array([[[mode(a)
                                               for a in a1]
                                              for a1 in a2]
                                             for a2 in attention_connection])
        else:
            attention_connection = None
        return layer_connection_pos / len(
            ind) * scaler, layer_connection_neg / len(
                ind) * scaler, attention_connection

    def make_graph(self,
                   pos,
                   ind,
                   tokens=None,
                   scaler=3,
                   show_legend = True,
                   plot_attention_connection=True,
                   width_scaler=.08):
        colors_bar = ['orange', 'purple', 'blue']
        colors = ['green', 'red']
        legend_threshold = 1
        color = 'black'
        dashes = [None, 'dash']
        layer_connection, layer_connection_neg, attention_connection = self.hm2layer_connection(
            ind, scaler, plot_attention_connection=plot_attention_connection)
        if attention_connection is None:
            attention_connection = np.zeros(
                (self.num_layer, self.num_word, self.num_word), dtype=np.int32)
            attention_connection_cnt = np.zeros(
                (self.num_layer, self.num_word, self.num_word), dtype=np.int32)
        else:
            attention_connection_cnt = attention_connection[:, :, :, 1]
            attention_connection = attention_connection[:, :, :, 0]
        layers = np.reshape(np.arange(self.num_word * (self.num_layer + 1)),
                            (self.num_layer + 1, self.num_word))
        layers = layers[:, ::-1]
        edges_x = []
        edges_y = []

        node_x = []
        node_y = []

        edge_width = []
        traces = []

        attention_node_x = []
        attention_node_y = []
        attention_node_n = []
        legend_flag = [False, False, False, False]
        legend_flag_track = [True, True, True, True]

        if not plot_attention_connection:
            legend_names = [
                'pos, skip ', 'attn baseline', 'neg, skip', 'neg, attn'
            ]
        else:
            legend_names = ['pos, skip', 'pos, attn', 'neg, skip', 'neg, attn']

        for k in range(len(layer_connection)):
            connection = layer_connection[k]
            for i in range(connection.shape[0]):
                for j in range(connection.shape[1]):
                    if connection[i][j] != 0:
                        val = connection[i][j]
                        if val > 0:
                            color = colors[0]
                        else:
                            color = colors[1]
                        source_node = layers[k][i]
                        target_node = layers[k + 1][j]
                        source_pos = pos[source_node]
                        target_pos = pos[target_node]
                        if source_pos[0] != target_pos[0]:
                            if attention_connection[k, i, j] == -1:
                                # if val < 0:
                                dash = dashes[1]
                                legend_idx = 0
                            else:
                                dash = dashes[0]
                                legend_idx = 1
                                if attention_connection_cnt[k, i,
                                                            j] > len(ind) / 4:
                                    attention_node_x.append(
                                        (source_pos[0] + target_pos[0]) / 2)
                                    attention_node_y.append(
                                        (source_pos[1] + target_pos[1]) / 2)
                                    attention_node_n.append(
                                        attention_connection[k, i, j] + 1)
                            if np.abs(val) > legend_threshold:
                                legend_flag[legend_idx] = True
                            else:
                                legend_flag[legend_idx] = False
                            edges_x = [source_pos[0], target_pos[0], ""]
                            edges_y = [source_pos[1], target_pos[1], ""]
                            # width = np.sqrt(np.abs(val)) *
                            #               sqrt_scaler
                            width = np.abs(val)
                            edge_trace = go.Scatter(
                                x=edges_x,
                                y=edges_y,
                                line=dict(width=width, color=color, dash=dash),
                                showlegend=legend_flag[legend_idx] &
                                legend_flag_track[legend_idx],
                                name=legend_names[legend_idx],
                                hoverinfo='text',
                                text="(" + str(source_node) + "," +
                                str(target_node) + ")",
                                opacity=0.8,
                                mode='lines')
                            if legend_flag[legend_idx]:
                                legend_flag_track[legend_idx] = False
                            traces.append(edge_trace)

        if layer_connection_neg is not None:
            color = colors[1]
            for k in range(len(layer_connection_neg)):
                connection = layer_connection_neg[k]
                for i in range(connection.shape[0]):
                    for j in range(connection.shape[1]):
                        if connection[i][j] != 0:
                            if attention_connection[k, i, j] == -1:
                                dash = dashes[1]
                                legend_idx = 2

                            else:
                                dash = dashes[0]
                                legend_idx = 3
                            val = connection[i][j]
                            source_node = layers[k][i]
                            target_node = layers[k + 1][j]

                            source_pos = pos[source_node]
                            target_pos = pos[target_node]

                            if source_pos[0] != target_pos[0]:

                                if attention_connection_cnt[k, i, j] > len(
                                        ind) / 3 and attention_connection[
                                            k, i, j] != -1:
                                    if not ((source_pos[0] + target_pos[0]) /
                                            2 in attention_node_x and
                                            (source_pos[1] + target_pos[1]) /
                                            2 in attention_node_y):
                                        attention_node_x.append(
                                            (source_pos[0] + target_pos[0]) / 2)
                                        attention_node_y.append(
                                            (source_pos[1] + target_pos[1]) / 2)
                                        attention_node_n.append(
                                            attention_connection[k, i, j] + 1)
                                if np.abs(val) > legend_threshold:
                                    legend_flag[legend_idx] = True
                                else:
                                    legend_flag[legend_idx] = False
                                edges_x = [source_pos[0], target_pos[0], ""]
                                edges_y = [
                                    source_pos[1] - 0.05, target_pos[1] - 0.05,
                                    ""
                                ]
                                width = np.abs(val)
                                edge_trace = go.Scatter(
                                    x=edges_x,
                                    y=edges_y,
                                    showlegend=legend_flag[legend_idx] &
                                    legend_flag_track[legend_idx],
                                    line=dict(width=width,
                                              color=color,
                                              dash=dash),
                                    name=legend_names[legend_idx],
                                    hoverinfo='text',
                                    text="(" + str(source_node) + "," +
                                    str(target_node) + ")",
                                    opacity=0.8,
                                    mode='lines')
                                if legend_flag[legend_idx]:
                                    legend_flag_track[legend_idx] = False

                                traces.append(edge_trace)

        node_x = []
        node_y = []
        for node in pos:
            x, y = pos[node][0], pos[node][1]
            node_x.append(x)
            node_y.append(y)

        if tokens is None:
            tokens = self.examples[ind][0].tokens

        node_trace_with_text = go.Scatter(x=node_x[:self.num_word],
                                          y=node_y[:self.num_word],
                                          showlegend=False,
                                          mode='markers+text',
                                          text=tokens,
                                          textposition='middle left',
                                          cliponaxis=False,
                                          textfont=dict(size=23),
                                          hoverinfo='text',
                                          marker=dict(showscale=False,
                                                      symbol='square',
                                                      opacity=0.7,
                                                      reversescale=True,
                                                      color='white',
                                                      size=20,
                                                      line=dict(color='black',
                                                                width=8),
                                                      line_width=2))

        node_trace_mask = go.Scatter(
            x=[node_x[-self.num_word + self.mask_index]],
            y=[node_y[-self.num_word + self.mask_index]],
            mode='markers',
            textposition='middle left',
            hoverinfo='text',
            showlegend=False,
            marker=dict(showscale=False,
                        symbol='circle',
                        opacity=0.7,
                        reversescale=True,
                        color='white',
                        size=20,
                        line=dict(color='black', width=8),
                        line_width=2))

        node_trace_last = go.Scatter(x=node_x[-self.num_word:],
                                     y=node_y[-self.num_word:],
                                     mode='markers',
                                     textposition='middle left',
                                     hoverinfo='text',
                                     showlegend=False,
                                     marker=dict(showscale=False,
                                                 symbol='circle',
                                                 opacity=0.3,
                                                 reversescale=True,
                                                 color='white',
                                                 size=10,
                                                 line=dict(color='black',
                                                           width=8),
                                                 line_width=2))

        node_trace = go.Scatter(x=node_x[self.num_word:-self.num_word],
                                y=node_y[self.num_word:-self.num_word],
                                mode='markers',
                                text=layers[1:].flatten(),
                                textposition='middle left',
                                hoverinfo='text',
                                showlegend=False,
                                marker=dict(showscale=False,
                                            symbol='circle',
                                            opacity=0.7,
                                            reversescale=True,
                                            color='white',
                                            size=10,
                                            line=dict(color='black', width=8),
                                            line_width=2))
        node_trace_attn = go.Scatter(
            x=attention_node_x,
            y=attention_node_y,
            mode='text',
            text=attention_node_n,
            textfont=dict(size=20),
            # textposition='bottom center',
            hoverinfo='text',
            showlegend=False,
            marker=dict(showscale=False,
                        symbol='circle',
                        opacity=1,
                        reversescale=True,
                        color='white',
                        size=24,
                        line=dict(color='black', width=8),
                        line_width=2))
        dis_scaler = 1
        base = 0.0

        emb_legend = '$\\large \mathcal{I}(\mathbf{x}_i, \pi_{baseline,i})$' if not plot_attention_connection else '$\\large \mathcal{I}(\mathbf{x}_i, \pi^e_i)$'

        dist_traces = []
        dist_traces.append(
            go.Bar(
                y=node_y[:self.num_word],
                x=np.flip(self.inf[ind].mean(0) * dis_scaler),
                width=np.ones_like(x) * width_scaler,
                base=base,
                name="$\\large g(\mathbf{x}_i)$",
                # name='g(x)',
                orientation='h',
                marker=dict(color=colors_bar[0], opacity=0.7),
                showlegend=plot_attention_connection,
                xaxis="x2",
                yaxis="y2"))
        dist_traces.append(
            go.Bar(y=node_y[:self.num_word],
                   x=np.flip(self.comb_greedy_val[ind].mean(0) * dis_scaler),
                   width=np.ones_like(x) * width_scaler,
                   base=base,
                   name=emb_legend,
                   marker=dict(color=colors_bar[1], opacity=0.7),
                   orientation='h',
                   xaxis="x2",
                   yaxis="y2"))
        dist_traces.append(
            go.Bar(y=node_y[:self.num_word],
                   x=np.flip(self.comb_greedy_attn[ind].mean(0) * dis_scaler),
                   width=np.ones_like(x) * width_scaler,
                   base=base,
                   name='$\\large \mathcal{I}(\mathbf{x}_i, \pi^a_i)$',
                   marker=dict(color=colors_bar[2], opacity=0.7),
                   orientation='h',
                   showlegend=plot_attention_connection,
                   xaxis="x2",
                   yaxis="y2"))
        data = traces + [
            node_trace_with_text, node_trace, node_trace_attn, node_trace_last,
            node_trace_mask
        ] + dist_traces
        if not plot_attention_connection:
            tickvals = [0., 0.02]
        else:
            tickvals = np.arange(-2, 2, 1)
        layout = go.Layout(
            template="simple_white",
            barmode='group',
            bargroupgap=0.0,
            bargap=0.4,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.9,
                font=dict(size=20, color="black"),
            ),
            showlegend=show_legend,
            margin=dict(b=40, l=5, r=5, t=10),
            xaxis=dict(domain=[0.25, .9],
                       automargin=True,
                       ticks='',
                       showline=False,
                       showticklabels=False,
                       showgrid=False,
                       zeroline=False),
            yaxis=dict(autorange='reversed',
                       automargin=True,
                       ticks='',
                       showline=False,
                       showticklabels=False,
                       showgrid=False,
                       zeroline=False),
            xaxis2=dict(
                domain=[0, 0.2],
                automargin=False,
                ticks='',
                range=[-4, 4],
                # range=[0, 1.5],
                # tickvals=np.arange(0, 1.1, 0.5),
                tickvals=tickvals,

                # tickvals=[0, 0.5, 1, 1.5],
                tickfont=dict(size=20),
                showline=True,
                showticklabels=True,
                showgrid=False,
                zeroline=True),
            yaxis2=dict(anchor="x2",
                        showline=False,
                        showticklabels=False,
                        ticks='',
                        showgrid=False,
                        zeroline=False),
            hovermode='closest',
            width=750,
            height=500)
        #large (950,500)
        #like (750,120)
        fig = go.Figure(data=data, layout=layout)
        return fig
