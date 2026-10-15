import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from crc_tools import discordant_pairs,kde_field,knn_place,loo_errors,neighbor_overlap,unit_rows
class CrcToolsTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(0);self.Z=unit_rows(rng.normal(size=(30,8)));self.xy=rng.normal(size=(30,2))*10
 def test_unit_rows_rejects_zero(self):
  with self.assertRaises(ValueError):unit_rows(np.zeros((2,3)))
 def test_identical_embedding_lands_near_its_twin(self):
  pos,nn,sim=knn_place(self.Z[4],self.Z,self.xy,k=1)
  self.assertEqual(nn[0],4);np.testing.assert_allclose(pos,self.xy[4])
 def test_exclude_removes_self(self):
  _,nn,_=knn_place(self.Z[4],self.Z,self.xy,k=5,exclude=4);self.assertNotIn(4,nn)
 def test_loo_is_finite(self):
  e=loo_errors(self.Z,self.xy,k=3);self.assertEqual(e.shape,(30,));self.assertTrue(np.isfinite(e).all())
 def test_discordant_pairs_sorted_and_cross_group(self):
  sim=self.Z@self.Z.T;linked=np.arange(30)<5
  i,j,s=discordant_pairs(sim,linked)
  self.assertEqual(len(s),5*25);self.assertTrue(linked[i].all() and not linked[j].any());self.assertTrue((np.diff(s)<=0).all())
 def test_neighbor_overlap_self_is_one(self):
  sim=self.Z@self.Z.T;self.assertAlmostEqual(neighbor_overlap(sim,sim,5),1.0)
 def test_kde_peak_normalised(self):
  g=np.c_[np.linspace(-5,5,50),np.zeros(50)];z=kde_field(np.array([[0.,0.]]),g,(1,50));self.assertAlmostEqual(z.max(),1.0,places=6)
if __name__=='__main__':unittest.main()
