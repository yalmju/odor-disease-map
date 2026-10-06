"""Validate fixed coordinates without silently repairing scientific data."""
import math

def validate(data):
    points=data.get('points',[])
    if len(points)<2:
        raise ValueError('At least two points required')
    by_id={}
    for p in points:
        key=p.get('index')
        if not isinstance(key,int) or key in by_id:
            raise ValueError('Unique integer index required')
        if not all(isinstance(p.get(k),(int,float)) and math.isfinite(p[k]) for k in ['x','y']):
            raise ValueError('Finite numeric coordinates required')
        for k in ['odors','diseases']:
            if not isinstance(p.get(k),list) or not all(isinstance(v,str) for v in p[k]):
                raise ValueError('Annotation lists of strings required')
        by_id[key]=p
    focal=data.get('focus',[])[:2]
    if len(focal)!=2 or focal[0]['index']==focal[1]['index']:
        raise ValueError('Two distinct focal points required')
    for p in focal:
        q=by_id.get(p['index'])
        if q is None or any(p.get(k)!=q.get(k) for k in ['x','y','odors','diseases']):
            raise ValueError('Focus coordinates and annotations must match the point table exactly')
    return data
