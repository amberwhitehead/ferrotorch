import torch, json, math

def rs(n, scale, offset):
    return [math.sin(i*0.137 + offset)*scale for i in range(n)]

def tl(t):
    return t.detach().contiguous().view(-1).tolist()

results = {}

def lstm_case(name, batch, seq, insz, hs, layers, with_state, loss_kind):
    m = torch.nn.LSTM(insz, hs, layers, batch_first=True, bias=True)
    sd = {}
    for l in range(layers):
        lin = insz if l == 0 else hs
        sd[f'weight_ih_l{l}'] = torch.tensor(rs(4*hs*lin, 0.3, 0.0+l)).reshape(4*hs, lin)
        sd[f'weight_hh_l{l}'] = torch.tensor(rs(4*hs*hs, 0.3, 1.7+l)).reshape(4*hs, hs)
        sd[f'bias_ih_l{l}'] = torch.tensor(rs(4*hs, 0.1, 0.5+l))
        sd[f'bias_hh_l{l}'] = torch.tensor(rs(4*hs, 0.1, 2.3+l))
    m.load_state_dict(sd, strict=True)
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.tensor(rs(batch*seq*insz, 0.5, 0.9)).reshape(batch, seq, insz).requires_grad_(True)
    state = None
    h0 = c0 = None
    if with_state:
        h0 = torch.tensor(rs(layers*batch*hs, 0.2, 3.1)).reshape(layers, batch, hs).requires_grad_(True)
        c0 = torch.tensor(rs(layers*batch*hs, 0.2, 4.4)).reshape(layers, batch, hs).requires_grad_(True)
        state = (h0, c0)
    out, (hn, cn) = m(x, state)
    if loss_kind == 'out':
        w = torch.tensor(rs(out.numel(), 1.0, 0.21)).reshape(out.shape)
        loss = (out*w).sum()
    else:
        w = torch.tensor(rs(hn.numel(), 1.0, 0.5)).reshape(hn.shape)
        loss = (hn*w).sum()
    loss.backward()
    r = {'out': tl(out), 'hn': tl(hn), 'cn': tl(cn), 'lossw': tl(w),
         'shape': [batch, seq, insz, hs, layers]}
    for l in range(layers):
        r[f'wih{l}'] = tl(getattr(m, f'weight_ih_l{l}').grad)
        r[f'whh{l}'] = tl(getattr(m, f'weight_hh_l{l}').grad)
        r[f'bih{l}'] = tl(getattr(m, f'bias_ih_l{l}').grad)
        r[f'bhh{l}'] = tl(getattr(m, f'bias_hh_l{l}').grad)
    r['xgrad'] = tl(x.grad)
    if with_state:
        r['h0grad'] = tl(h0.grad)
        r['c0grad'] = tl(c0.grad)
    results[name] = r

def gru_case(name, batch, seq, insz, hs, layers, with_state, loss_kind):
    m = torch.nn.GRU(insz, hs, layers, batch_first=True, bias=True)
    sd = {}
    for l in range(layers):
        lin = insz if l == 0 else hs
        sd[f'weight_ih_l{l}'] = torch.tensor(rs(3*hs*lin, 0.3, 0.0+l)).reshape(3*hs, lin)
        sd[f'weight_hh_l{l}'] = torch.tensor(rs(3*hs*hs, 0.3, 1.7+l)).reshape(3*hs, hs)
        sd[f'bias_ih_l{l}'] = torch.tensor(rs(3*hs, 0.1, 0.5+l))
        sd[f'bias_hh_l{l}'] = torch.tensor(rs(3*hs, 0.1, 2.3+l))
    m.load_state_dict(sd, strict=True)
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.tensor(rs(batch*seq*insz, 0.5, 0.9)).reshape(batch, seq, insz).requires_grad_(True)
    state = None
    h0 = None
    if with_state:
        h0 = torch.tensor(rs(layers*batch*hs, 0.2, 3.1)).reshape(layers, batch, hs).requires_grad_(True)
        state = h0
    out, hn = m(x, state)
    if loss_kind == 'out':
        w = torch.tensor(rs(out.numel(), 1.0, 0.21)).reshape(out.shape)
        loss = (out*w).sum()
    else:
        w = torch.tensor(rs(hn.numel(), 1.0, 0.5)).reshape(hn.shape)
        loss = (hn*w).sum()
    loss.backward()
    r = {'out': tl(out), 'hn': tl(hn), 'lossw': tl(w),
         'shape': [batch, seq, insz, hs, layers]}
    for l in range(layers):
        r[f'wih{l}'] = tl(getattr(m, f'weight_ih_l{l}').grad)
        r[f'whh{l}'] = tl(getattr(m, f'weight_hh_l{l}').grad)
        r[f'bih{l}'] = tl(getattr(m, f'bias_ih_l{l}').grad)
        r[f'bhh{l}'] = tl(getattr(m, f'bias_hh_l{l}').grad)
    r['xgrad'] = tl(x.grad)
    if with_state:
        r['h0grad'] = tl(h0.grad)
    results[name] = r

def rnn_case(name, batch, seq, insz, hs, layers, with_state, nonlin, loss_kind):
    m = torch.nn.RNN(insz, hs, layers, nonlinearity=nonlin, batch_first=True, bias=True)
    sd = {}
    for l in range(layers):
        lin = insz if l == 0 else hs
        sd[f'weight_ih_l{l}'] = torch.tensor(rs(hs*lin, 0.3, 0.0+l)).reshape(hs, lin)
        sd[f'weight_hh_l{l}'] = torch.tensor(rs(hs*hs, 0.3, 1.7+l)).reshape(hs, hs)
        sd[f'bias_ih_l{l}'] = torch.tensor(rs(hs, 0.1, 0.5+l))
        sd[f'bias_hh_l{l}'] = torch.tensor(rs(hs, 0.1, 2.3+l))
    m.load_state_dict(sd, strict=True)
    for p in m.parameters():
        p.requires_grad_(True)
    x = torch.tensor(rs(batch*seq*insz, 0.5, 0.9)).reshape(batch, seq, insz).requires_grad_(True)
    state = None
    h0 = None
    if with_state:
        h0 = torch.tensor(rs(layers*batch*hs, 0.2, 3.1)).reshape(layers, batch, hs).requires_grad_(True)
        state = h0
    out, hn = m(x, state)
    if loss_kind == 'out':
        w = torch.tensor(rs(out.numel(), 1.0, 0.21)).reshape(out.shape)
        loss = (out*w).sum()
    else:
        w = torch.tensor(rs(hn.numel(), 1.0, 0.5)).reshape(hn.shape)
        loss = (hn*w).sum()
    loss.backward()
    r = {'out': tl(out), 'hn': tl(hn), 'lossw': tl(w),
         'shape': [batch, seq, insz, hs, layers]}
    for l in range(layers):
        r[f'wih{l}'] = tl(getattr(m, f'weight_ih_l{l}').grad)
        r[f'whh{l}'] = tl(getattr(m, f'weight_hh_l{l}').grad)
        r[f'bih{l}'] = tl(getattr(m, f'bias_ih_l{l}').grad)
        r[f'bhh{l}'] = tl(getattr(m, f'bias_hh_l{l}').grad)
    r['xgrad'] = tl(x.grad)
    if with_state:
        r['h0grad'] = tl(h0.grad)
    results[name] = r

lstm_case('lstm_state', 3, 5, 4, 6, 1, True, 'out')
lstm_case('lstm_nostate_long', 2, 8, 3, 5, 1, False, 'hn')
lstm_case('lstm_2layer', 2, 4, 3, 4, 2, True, 'out')
lstm_case('lstm_seq1', 3, 1, 4, 5, 1, True, 'out')
gru_case('gru_state', 3, 5, 4, 6, 1, True, 'out')
gru_case('gru_2layer', 2, 4, 3, 4, 2, True, 'out')
gru_case('gru_seq1', 3, 1, 4, 5, 1, False, 'out')
rnn_case('rnn_tanh_state', 3, 5, 4, 6, 1, True, 'tanh', 'out')
rnn_case('rnn_relu_state', 3, 5, 4, 6, 1, True, 'relu', 'out')
rnn_case('rnn_tanh_2layer', 2, 4, 3, 4, 2, True, 'tanh', 'out')

open('/tmp/rnn_oracle.json', 'w').write(json.dumps(results))
