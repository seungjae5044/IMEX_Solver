import torch 

class Solver:
    def __init__(self, alpha, alphas, alpha_bars):
        self.alpha = alpha
        self.alphas = alphas
        self.alpha_bars = alpha_bars


    def inverse_solver(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1
        alpha = self.alphas[t_idx]
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx-1]

        N = alpha.size(0)
        alpha = alpha.view(N,1,1,1)
        alpha_bar = alpha_bar.view(N,1,1,1)
        alpha_bar_prev = alpha_bar_prev.view(N,1,1,1)

        model.eval()
        with torch.no_grad():
            eps = model(x, t)
        model.train()

        noise = torch.randn_like(x, device = self.device)
        noise[t == 1] = 0

        mu = (x - ((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std = torch.sqrt((1-alpha) * (1-alpha_bar_prev) / 1-alpha_bar)
        return mu + noise * std