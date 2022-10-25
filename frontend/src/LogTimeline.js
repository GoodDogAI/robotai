import { useEffect, useState } from "react";
import { useQuery } from "react-query";
import { FixedSizeList as List } from 'react-window';

import axios from "axios";


function getMessageType(msg) {
    const { valid, logMonoTime, _total_size_bytes, ...rest } = msg;
    const mainKey = Object.keys(rest).pop();

    return mainKey;
}

function formatSize(bytes) {
    if (bytes < 1024)
        return `${bytes}B`;
    else
        return `${(bytes/1024).toFixed(1)}kB`;
}

function FrameSlider(props) {
    const { allFrameIds, index, onChangeIndex } = props;

    return (
        <input className="frameSlider" type="range" min="0" max={allFrameIds.length} value={index} onChange={(evt) => onChangeIndex(evt.target.value)} />
    );
}

export function LogTimeline(props) {
    const { logName } = props;

    const { isLoading, error, data } = useQuery(["logs", logName], () =>
        axios.get(
          `${process.env.REACT_APP_BACKEND_URL}/logs/${logName}`
        ).then((res) => res.data),
        {
            enabled: !!logName,
        }
      );

    const [logIndex, setLogIndex] = useState(0);

    // Reset the frame index back to zero if the logname changes
    useEffect(() => {
        setLogIndex(0);
    }, [logName]);




    if (!logName) {
        return <div className="timeline">Select a log to view</div>;
    }

    if (error) {
        return <div className="timeline">Error loading {error}</div>;
    }

    if (isLoading) {
        return <div className="timeline">Loading...</div>;
    }

    const frameIds = [], allFrameIds = [-1];
    for (const message of data) {
        if ("headEncodeData" in message){
            frameIds.push(message["headEncodeData"]["idx"]["frameId"]);
            allFrameIds.push(message["headEncodeData"]["idx"]["frameId"]);
        }
        else {
            allFrameIds.push(allFrameIds[allFrameIds.length - 1]);
        }
    }

    const Row = ({ index, style }) => (
    <div className="row" style={style}>
               <div className="cell index">{data[index].index}</div>
               <div className="cell which">{data[index].which}</div>
               <div className="cell size">{formatSize(data[index].total_size_bytes)}</div>
           </div>
);

    return (
        <div className="timeline">
            <div className="frameContainer">
                <div style={{position: "relative"}}>
                    <img width="100%" src={`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/frame/${allFrameIds[logIndex]}`} alt={`frame${allFrameIds[logIndex]}`}   style={{position: "absolute", zIndex: 0}} />
                    <img width="100%" src={`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/frame_reward/${allFrameIds[logIndex]}`} alt={`reward${allFrameIds[logIndex]}`} style={{position: "relative", zIndex: 1}} />
                </div>
                <div>
                    <span>Frame {allFrameIds[logIndex]} / {frameIds.length} (ID{allFrameIds[logIndex]})</span>
                    <FrameSlider allFrameIds={allFrameIds} index={logIndex} onChangeIndex={setLogIndex}/>
                </div>
            </div>
            <h5>{logName}</h5>
            <div className="logTable">

                  <List
    height={500}
    itemCount={data.length}
    itemSize={25}
    width={500}
  >
    {Row}
  </List>

            </div>
        </div>
    );
}