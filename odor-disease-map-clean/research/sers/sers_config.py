"""Shared band definitions for the reporter-channel SERS analysis.

Channels: pristine (no reporter), mba (4-mercaptobenzoic acid), mpy (4-mercaptopyridine),
nat (2-naphthalenethiol). Ratios are (numerator band, reference band) in cm-1; the
reference band is the channel's strongest reporter band so every feature is independent
of absolute intensity. Pristine has no reporter; its reference is the 1340 indole band.
"""
CHANNELS = ['pristine', 'mba', 'mpy', 'nat']
ANALYTES = ['blank', 'ind', 'skt', 'mix']
REF_BAND = {'pristine': None, 'mba': 1078, 'mpy': 1095, 'nat': 1064}
RATIOS = {
    'pristine': [(1010, 1340), (1490, 1340), (1580, 1340), (760, 1340), (1220, 1340)],
    'mba': [(1078, 1590), (1420, 1078), (1700, 1078), (1010, 1078), (1340, 1078)],
    'mpy': [(1004, 1095), (1610, 1580), (1060, 1095), (1218, 1095), (1340, 1095)],
    'nat': [(1380, 1064), (1580, 1064), (1623, 1064), (1340, 1064), (760, 1064)],
}
# two ratios per channel used for the 2-D scatter panel
PAIR = {'pristine': ((1580, 1340), (1490, 1340)), 'mba': ((1340, 1078), (1700, 1078)),
        'mpy': ((1610, 1580), (1060, 1095)), 'nat': ((1380, 1064), (1580, 1064))}
LABEL = {'blank': 'Reporter only', 'ind': 'Indole', 'skt': 'Skatole', 'mix': 'Indole + skatole'}
COLOR = {'blank': '#8c8c8c', 'ind': '#b78ee6', 'skt': '#d2608a', 'mix': '#3aa17e'}
