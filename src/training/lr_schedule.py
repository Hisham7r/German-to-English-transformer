
def get_warmup_lr_lambda(d_model, warmup_steps):
        """
        Returns a function that computes the learning rate at a given step,
        following the schedule from Section 5.3 of "Attention Is All You Need".

        Formula:
            lr = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
        """
        def lr_lambda(step):
            step = max(step, 1)
            a = step ** -0.5
            b = step * warmup_steps ** -1.5
            return (d_model ** -0.5) * min(a, b)
        return lr_lambda

    # ── Test-Block ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
        d_model = 512
        warmup_steps = 4000
        lr_fn = get_warmup_lr_lambda(d_model, warmup_steps)
        checkpoints = [1, 500, 1000, 2000, 3000, 4000, 6000, 10000, 16000, 50000]
        print(f"{'step':>8} | {'learning rate':>15}")
        print("-" * 28)
        for step in checkpoints:
            print(f"{step:>8} | {lr_fn(step):>15.8f}")
