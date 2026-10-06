import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from validate_data import validate
class ValidationTests(unittest.TestCase):
 def setUp(self): self.d=json.loads((ROOT/'data/demo.json').read_text())
 def test_valid(self): validate(self.d)
 def test_nonfinite(self):
  self.d['points'][2]['x']=float('nan')
  with self.assertRaises(ValueError):validate(self.d)
 def test_duplicate(self):
  self.d['points'][2]['index']=self.d['points'][3]['index']
  with self.assertRaises(ValueError):validate(self.d)
 def test_stale_focus(self):
  self.d['focus'][0]['x']+=1
  with self.assertRaises(ValueError):validate(self.d)
 def test_changed_membership(self):
  self.d['focus'][0]['diseases']=['incorrect association']
  with self.assertRaises(ValueError):validate(self.d)
if __name__=='__main__':unittest.main()
