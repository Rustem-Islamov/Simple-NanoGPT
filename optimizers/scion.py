import torch

@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError


class Spectral(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        d_out, d_in = g.shape
        g *= (d_out / d_in)**0.5
        return g


class Sign(Norm):
    def lmo(self, g):
        out_channels, in_channels = g.shape
        return (1 / in_channels) * torch.sign(g)


norm_dict = {
    'Spectral': Spectral,
    'Sign': Sign,
}


class Scion(torch.optim.Optimizer):
    """
    Scion (https://arxiv.org/abs/2502.07529): a norm-constrained Frank-Wolfe-style optimizer.

    Each parameter group picks a norm ('Spectral' for matrices, 'Sign' for the
    embedding/head) whose linear minimization oracle (`lmo`) turns the (momentum-averaged)
    gradient into a update direction of a fixed norm, scaled by `scale`. 'momentum' is the
    dual-averaging weight alpha in m_{k+1} = (1 - alpha) m_k + alpha g_k. 'lr' is the step
    size, normally driven by an external LR scheduler. With `unconstrained=False` (default)
    the parameter is kept inside the `scale`-ball via a convex combination with the update;
    with `unconstrained=True` the update is a plain additive step.
    """
    def __init__(self, params, lr=1e-3, momentum=1.0, norm: str = 'Spectral',
                 norm_kwargs: dict = None, scale=1.0, unconstrained=False, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, norm=norm, norm_kwargs=norm_kwargs or {},
                         scale=scale, unconstrained=unconstrained, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            weight_decay = group['weight_decay']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                # momentum buffer
                if momentum != 1:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                # normalized update
                update = scale * norm_backend.lmo(g)

                # apply weight decay decoupled from gradient
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # apply update
                if unconstrained:
                    p.data.add_(update, alpha=-lr)
                else:
                    p.data.mul_(1 - lr).add_(update, alpha=-lr)
