"""Small synthetic interface checks; no paper result is recomputed."""
import ast,copy,json,sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from torch_geometric.data import Data
from lrr.models.gcn import GCNBackbone
from lrr.models.gat import GATBackbone
from training.train_gcn import prepare_gcn,_update as update_gcn
from training.train_gat import prepare_gat,_update as update_gat
from build_bundle import homogeneous
from pretrain import train


class Pipeline(unittest.TestCase):
    def fixture(self,family):
        torch.manual_seed(42);n=60
        edges=torch.combinations(torch.arange(n),r=2).t().contiguous()
        data=Data(x=torch.rand(n,6),num_nodes=n,train_pos_edge_index=edges[:,:240],
            val_pos_edge_index=edges[:,240:250],test_pos_edge_index=edges[:,250:260],
            val_neg_edge_index=edges[:,240:250].flip(0),test_neg_edge_index=edges[:,250:260].flip(0))
        cfg=dict(dataset='DBLP',backbone=family.upper(),deletion_mode='Ed_IN',deletion_ratio='2.5',seed=42,
            updates=1,validation_interval=1,lr=.0003,weight_decay=0.,sampling_identity=dict(dataset='DBLP',backbone=family,deletion_mode='in',deletion_ratio='2.5',run_seed=42))
        state=(GCNBackbone(6) if family=='gcn' else GATBackbone(6)).state_dict()
        return data,cfg,state

    def test_source_syntax(self):
        for p in ROOT.rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))

    def test_prepare_and_one_update(self):
        for family in ['gcn','gat']:
            with self.subTest(backbone=family):
                data,cfg,state=self.fixture(family);bundle=homogeneous(cfg,data,state)
                prepared=prepare_gcn(cfg,bundle) if family=='gcn' else prepare_gat(cfg,bundle)
                if family=='gcn':parts=update_gcn(prepared,0,capture=True)
                else:parts=update_gat(prepared,0,capture=True)
                self.assertTrue(bool(torch.isfinite(parts['total'])))
                self.assertEqual(sum(p.numel() for p in prepared.model.parameters() if p.requires_grad),20480)
                self.assertTrue(all(p.grad is None for p in prepared.model.backbone.parameters()))

    def test_pretrain_interfaces(self):
        for family in ['gcn','gat']:
            data,cfg,_=self.fixture(family)
            cfg.update(backbone=family,epochs=1,validation_interval=1,lr=.001,
                       selector='max_validation_dt_auc' if family=='gcn' else 'min_validation_bce',
                       model_seed=123,training_negative_seed=456)
            result=train(cfg,data,torch.device('cpu'),1)
            self.assertIsInstance(result['model_state'],dict)
            self.assertFalse(result['test_used_for_selection'])

    def test_published_evaluation_interface(self):
        from evaluate_published import evaluate
        from training.train_gcn import evaluate_gcn_test
        from training.train_gat import evaluate_gat_test
        for family in ['gcn','gat']:
            data,cfg,state=self.fixture(family);bundle=homogeneous(cfg,data,state)
            prepared=prepare_gcn(cfg,bundle) if family=='gcn' else prepare_gat(cfg,bundle)
            expected=evaluate_gcn_test(prepared,bundle['test']) if family=='gcn' else evaluate_gat_test(prepared,bundle['test'])
            actual=evaluate(prepared.model,bundle,family)
            for key in ['DT_AUC','DT_AUP','Df_AUC','Df_AUP','J_AUC','J_AUP']:
                self.assertAlmostEqual(actual[key],expected[key],places=6)

    def test_rgcn_pretrain_interface(self):
        data,cfg,_=self.fixture('gcn');n=data.num_nodes
        train=data.train_pos_edge_index[:,:160];types=torch.arange(train.size(1))%11
        data=Data(x=torch.arange(n),num_nodes=n,edge_index=torch.cat((train,train.flip(0)),dim=1),
                  edge_type=torch.cat((types,types+11)),train_pos_edge_index=train,train_edge_type=types,
                  valid_pos_edge_index=train[:,:10],valid_edge_type=types[:10],
                  test_pos_edge_index=train[:,10:20],test_edge_type=types[10:20])
        cfg.update(backbone='rgcn',epochs=1,validation_interval=1,lr=.001,selector='max_validation_dt_aup',
                   training=dict(graphsaint_batch_size=8,graphsaint_walk_length=2,graphsaint_num_steps=1))
        result=train_rgcn_smoke(cfg,data)
        self.assertIsInstance(result['model_state'],dict)

    def test_published_request_preserves_source_order(self):
        import hashlib,tempfile,numpy as np
        from unittest.mock import patch
        import build_bundle
        data,cfg,state=self.fixture('gat');cfg['source_cell_id']='ordered_fixture'
        expected=data.train_pos_edge_index[:,[20,1,16,9,41,37]]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'data/deletion_requests/ordered_fixture.npz';p.parent.mkdir(parents=True)
            np.savez_compressed(p,df=expected.numpy());(root/'manifests').mkdir()
            (root/'manifests/deletion_requests.json').write_text(json.dumps([dict(cell_id='ordered_fixture',sha256=hashlib.sha256(p.read_bytes()).hexdigest())]))
            with patch.object(build_bundle,'ROOT',root):bundle=homogeneous(cfg,data,state)
        self.assertTrue(torch.equal(bundle['training']['df'],expected))

    def test_published_rgcn_evaluation_interface(self):
        from evaluate_published import evaluate
        from lrr.models.rgcn import RGCNBackbone,FullRankRGCN
        data,_,_=self.fixture('gcn');train=data.train_pos_edge_index
        n=data.num_nodes;typ=torch.arange(train.size(1))%11
        bank=dict(x=torch.arange(n),edge_index=torch.cat((train,train.flip(0)),dim=1),edge_type=torch.cat((typ,typ+11)),
                  gate1=torch.ones(n,1),gate2=torch.ones(n,1),val_positive=data.test_pos_edge_index,
                  val_negative=data.test_neg_edge_index,val_relation=torch.zeros(data.test_pos_edge_index.size(1),dtype=torch.long),
                  requested=train[:,:2],requested_relation=typ[:2],retained_control_groups=train[:,20:22].unsqueeze(0).repeat(500,1,1),
                  retained_control_relations=typ[20:22].unsqueeze(0).repeat(500,1))
        result=evaluate(FullRankRGCN(RGCNBackbone(n,11)),{'test':bank},'rgcn')
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
        self.assertAlmostEqual(result['J_AUC'],(result['DT_AUC']+result['Df_AUC'])/2)


def train_rgcn_smoke(config,data):
    return train(config,data,torch.device('cpu'),1)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
