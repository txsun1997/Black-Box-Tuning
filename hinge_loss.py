import torch

REDUCE_FN_MAPPINGS = {
    'sum': torch.sum,
    'mean': torch.mean,
    'none': lambda x: x
}

def hinge_loss(logit, target, margin, reduce='sum'):
    """
    Args:
        logit (torch.Tensor): (N, C, d_1, d_2, ..., d_K)
        target (torch.Tensor): (N, d_1, d_2, ..., d_K)
        margin (float):
    """
    target = target.unsqueeze(1)
    tgt_logit = torch.gather(logit, dim=1, index=target)
    loss = logit - tgt_logit + margin
    loss = torch.masked_fill(loss, loss < 0, 0)
    loss = torch.scatter(loss, dim=1, index=target, value=0)
    reduce_fn = REDUCE_FN_MAPPINGS[reduce]
    return reduce_fn(loss)


if __name__ == '__main__':
    # torch.manual_seed(100)
    bsz = 128
    n_labels = 10
    margin = 1.5

    def raw_hinge(logit, target, margin):
        loss = 0
        pred = logit.cpu().numpy().tolist()
        for i in range(len(target)):
            tgt = target[i].item()
            prd = pred[i]
            for j, p in enumerate(prd):
                if j == tgt:
                    continue
                else:
                    tmp = p - prd[tgt] + margin
                    if tmp > 0:
                        loss += tmp
        return loss

    logit = torch.rand(size=(bsz, n_labels))
    target = torch.randint(n_labels, size=(bsz,))
    print(logit)
    print(target)
    value1 = hinge_loss(logit, target, margin)
    value2 = raw_hinge(logit, target, margin)
    print(value1, value2, value1-value2)
