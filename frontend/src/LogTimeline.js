import { useEffect, useState, useRef } from "react";
import { useQuery } from "react-query";
import { VariableSizeList as List } from 'react-window';
import classNames from 'classnames';
import axios from "axios";


function formatSize(bytes) {
    if (bytes < 1024)
        return `${bytes}B`;
    else
        return `${(bytes / 1024).toFixed(1)}kB`;
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

    const { data: logEntryData } = useQuery(["logs", logName, logIndex], () =>
        axios.get(
            `${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/entry/${logIndex}`
        ).then((res) => res.data),
        {
            enabled: !!logName,
        }
    );

    const listEl = useRef(null);

    useEffect(() => {
        if (listEl.current) {
            listEl.current.resetAfterIndex(0);
        }
    }, [logIndex, logEntryData]);

    if (!logName) {
        return <div className="timeline">Select a log to view</div>;
    }

    if (error) {
        return <div className="timeline">Error loading {error}</div>;
    }

    if (isLoading) {
        return <div className="timeline">Loading...</div>;
    }

    const frameIds = new Set(data.map(entry => entry.headIndex));

    const Row = ({ index, style }) => {
        const rowClass = classNames({
            "row": true,
            "selected": index === logIndex,
        });

        let rowData = null;

        if (index === logIndex) {
            rowData = <pre>{JSON.stringify(logEntryData, null, 2)}</pre>;
        }

        return (
            <div className={rowClass} style={style} onClick={() => setLogIndex(index)}>
                <div className="cell index">{data[index].index}</div>
                <div className="cell which">{data[index].which}</div>
                <div className="cell size">{formatSize(data[index].total_size_bytes)}</div>
                <div>
                    {rowData}
                </div>
            </div>
        );
    }

    const GetRowSize = (index) => {
        if (index === logIndex && logEntryData) {
            return 20 * JSON.stringify(logEntryData, null, 2).split(/\r\n|\r|\n/).length;
        }
        return 25;
    }

    return (
        <div className="timeline">
            <div className="frameContainer">
                <div style={{ position: "relative" }}>
                    <img width="100%" src={`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/frame/${data[logIndex].headIndex}`} alt={`frame${data[logIndex].headIndex}`} style={{ position: "absolute", zIndex: 0 }} />
                    <img width="100%" src={`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/frame_reward/${data[logIndex].headIndex}`} alt={`reward${data[logIndex].headIndex}`} style={{ position: "relative", zIndex: 1 }} />
                </div>
                {/* <div>
                    <span>Frame {allFrameIds[logIndex]} / {frameIds.length} (ID{allFrameIds[logIndex]})</span>
                    <FrameSlider allFrameIds={allFrameIds} index={logIndex} onChangeIndex={setLogIndex} />
                </div> */}
            </div>
            <h5>{logName}</h5>
            <div className="logTable">
                <div className="row header">
                    <div className="cell index">Index</div>
                    <div className="cell which">Which</div>
                    <div className="cell size">Size</div>
                </div>
                <List
                    ref={listEl}
                    height={800}
                    itemCount={data.length}
                    itemSize={GetRowSize}
                    width={800}>

                    {Row}
                </List>

            </div>
        </div>
    );
}