import copy
import torch
from torch import nn
from src.kronos_continual.lora_suffix import LoRALinear,ranking_loss,objective


def test_lora_zero_start_gradients_and_frozen_weight():
    torch.manual_seed(7)
    linear=nn.Linear(12,8); original=copy.deepcopy(linear.state_dict()); layer=LoRALinear(linear,4,8)
    x=torch.randn(16,12);target=torch.randn(16,8)
    torch.testing.assert_close(layer(x),linear(x),rtol=0,atol=0)
    (layer(x)-target).square().mean().backward()
    assert layer.b.grad.norm()>0 and layer.a.grad.norm()==0
    with torch.no_grad(): layer.b.add_(-.01*layer.b.grad)
    layer.zero_grad(); (layer(x)-target).square().mean().backward()
    assert layer.a.grad.norm()>0 and layer.base.weight.grad is None
    for key,value in original.items(): torch.testing.assert_close(value,layer.base.state_dict()[key],rtol=0,atol=0)


def test_rank_loss_orders_and_ties():
    y=torch.arange(5,dtype=torch.float32)[:,None].repeat(1,10)
    assert ranking_loss(y,y,1)<ranking_loss(-y,y,1)
    pred=torch.randn(5,10,requires_grad=True)
    tied=torch.zeros(5,10)
    loss=ranking_loss(pred,tied,1);loss.backward()
    assert torch.equal(pred.grad,torch.zeros_like(pred))
    perm=torch.tensor([4,1,2,0,3]);torch.testing.assert_close(ranking_loss(y,y,1),ranking_loss(y[perm],y[perm],1))


def test_objective_backward_scale():
    p=torch.randn(16,10,requires_grad=True);y=torch.randn(16,10)
    loss,path,rank=objective(p,y,torch.ones(10),1,.1);loss.backward()
    assert torch.isfinite(p.grad).all() and path>0 and rank>0


def test_actual_transformer_suffix_identity_and_freezing():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/Kronos_official_src/Kronos-master'))
    from model.module import TransformerBlock,RMSNorm
    from src.kronos_continual.lora_suffix import KronosSuffix
    from src.kronos_continual.attention_adapter import Adapter
    torch.set_num_threads(1);torch.manual_seed(5)
    head=nn.Module();head.adapter=Adapter(16,4,0);head.head=nn.Linear(16,10)
    blocks=[TransformerBlock(16,2,32,0,0,0) for _ in range(2)];norm=RMSNorm(16)
    original=KronosSuffix(copy.deepcopy(blocks),copy.deepcopy(norm),copy.deepcopy(head),False).eval()
    adapted=KronosSuffix(copy.deepcopy(blocks),copy.deepcopy(norm),copy.deepcopy(head),True).eval()
    x=torch.randn(8,9,16);target=torch.randn(8,10)
    torch.testing.assert_close(original(x),adapted(x),rtol=0,atol=0)
    frozen={n:p.detach().clone() for n,p in adapted.named_parameters() if not p.requires_grad}
    trainable=[p for p in adapted.parameters() if p.requires_grad]
    for _ in range(2):
        adapted.zero_grad();(adapted(x)-target).square().mean().backward()
        with torch.no_grad():
            for p in trainable:p.add_(-.01*p.grad)
    for n,p in adapted.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
            assert p.grad is None
    assert adapted.blocks[0].self_attn.q_proj.a.grad.norm()>0
    assert not torch.equal(original(x),adapted(x))
