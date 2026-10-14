import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'gad/.backends/gad'))
from test_online_rubric import pair


class BatchTests(unittest.TestCase):
    def batch(self):
        import torch,numpy as np
        from verl import DataProto
        response=torch.tensor([[5,6,0,0],[7,8,0,0]])
        ids=torch.cat([torch.tensor([[0,1,2],[0,1,2]]),response],-1)
        mask=torch.tensor([[0,1,1,1,1,0,0]]*2)
        pos=torch.tensor([[0,0,1,2,3,4,5]]*2)
        teacher=torch.tensor([[3,4,0,0]]*2)
        tensors={'prompts':ids[:,:3],'input_ids':ids,'responses':response,'attention_mask':mask,'position_ids':pos,
                 'teacher_response':teacher,'teacher_input_ids':torch.cat([ids[:,:3],teacher],-1),
                 'teacher_attention_mask':mask.clone(),'teacher_position_ids':pos.clone(),
                 'advantages':torch.ones(2,4),'old_log_probs':torch.ones(2,4)}
        return DataProto.from_dict(tensors=tensors,non_tensors={'uid':np.array(['u1','u2'],dtype=object),
               'reward_model':np.array([{'ground_truth':'2'}]*2,dtype=object),
               'extra_info':np.array([{'index':1},{'index':1}],dtype=object),
               'gad_finish_reason':np.array(['stop','length'],dtype=object)},meta_info={'gad_pad_token_id':0})

    def test_exact_tokens_actor_immutable_and_finish(self):
        import torch
        from gad.extensions.critic_batch import extract_pairs,pack_pairs
        class Tokenizer:
            def decode(self,ids,**kwargs):return ','.join(map(str,ids))
        actor=self.batch();before={k:v.clone() for k,v in actor.batch.items()}
        records=extract_pairs(actor,Tokenizer(),'token_version',1,'gad')
        critic=pack_pairs(list(reversed(records)),actor)
        self.assertTrue(records[1]['student']['truncated'])
        self.assertEqual(records[0]['prompt_id'],records[1]['prompt_id'])
        self.assertNotEqual(records[0]['student']['response_id'],records[1]['student']['response_id'])
        self.assertEqual([7,8],critic.batch['responses'][0,:2].tolist())
        self.assertNotIn('advantages',critic.batch)
        self.assertNotIn('old_log_probs',critic.batch)
        for key,value in before.items():self.assertTrue(torch.equal(value,actor.batch[key]))
        self.assertEqual(['u1','u2'],actor.non_tensor_batch['uid'].tolist())
        self.assertEqual(critic.meta_info['global_token_num'],critic.batch['attention_mask'].sum(-1).tolist())

    def test_full_sequence_not_retokenized(self):
        from gad.extensions.critic_batch import pack_pairs
        record=pair(1);actor=self.batch()
        record['student']['token_ids']=[31,32,33,34]
        rebuilt=pack_pairs([record,record],actor)
        self.assertEqual([31,32,33,34],rebuilt.batch['responses'][0].tolist())
        record['student']['token_ids'].append(35)
        with self.assertRaises(ValueError):pack_pairs([record,record],actor)


if __name__=='__main__':unittest.main()
