from __future__ import annotations
import numpy as np

def trimmed_mean(scores:list[float],trim_fraction:float=0.10)->tuple[float,int]:
    """Deterministic trimmed mean.

    trim_count = floor(n * trim_fraction); the lowest and highest trim_count
    values are removed only when 2*trim_count < n and trim_count > 0. With four
    sampled frames trim_count is 0, so this reduces to the plain mean.
    """
    values=np.sort(np.asarray(scores,dtype=np.float64)); count=values.size
    if count==0: raise ValueError("cannot aggregate an empty video")
    trim=int(np.floor(count*trim_fraction))
    if trim>0 and 2*trim<count: return float(values[trim:count-trim].mean()),trim
    return float(values.mean()),0
def aggregate_videos(rows:list[dict],threshold:float,*,trim_fraction:float=0.10,key:str="source_record_id",
                     score_key:str="p_spoof_calibrated")->list[dict]:
    """Group frame predictions by safe record identifier; no label required."""
    grouped:dict[str,list[dict]]={}
    for row in rows: grouped.setdefault(str(row[key]),[]).append(row)
    aggregates=[]
    for record_id in sorted(grouped):
        frames=sorted(grouped[record_id],key=lambda row:row["sample_id"])
        score,trim=trimmed_mean([float(frame[score_key]) for frame in frames],trim_fraction)
        confidence=float(np.median([float(frame["confidence"]) for frame in frames]))
        aggregates.append({key:record_id,"frames":len(frames),"trim_count":trim,"video_score":score,
                           "video_confidence":confidence,"decision":"spoof" if score>=threshold else "live"})
    return aggregates
