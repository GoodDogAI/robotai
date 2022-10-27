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

    const GetNextLogIndex = (index) => {
        for (let i = index + 1; i < data.length; i++) {
            if (data[i].headIndex !== data[index].headIndex) {
                return i;
            }
        }

        return data.length - 1;
    }

    const GetPrevLogIndex = (index) => {
        for (let i = index - 1; i >= 0; i--) {
            if (data[i].headIndex !== data[index].headIndex) {
                return i;
            }
        }

        return 0;
    }


    return (
        <div className="timeline">
            <div className="frameContainer">
                <div style={{ position: "relative", top: 0, left: 0 }}>
                    <img width="100%" src={`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/frame/${data[logIndex].headIndex}`} alt={`frame${data[logIndex].headIndex}`} style={{ position: "relative", top:0, left:0, zIndex: 0 }} />
                    <img width="100%" src={`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/frame_reward/${data[logIndex].headIndex}`} alt={`reward${data[logIndex].headIndex}`} style={{ position: "absolute", top:0, left:0, zIndex: 1 }} />
                </div>

                Frame {data[logIndex].headIndex}

                <button type="button" onClick={() => setLogIndex(Array(30).fill(0).reduce((index, _)  => GetPrevLogIndex(index), logIndex))} disabled={logIndex === 0}>-30</button>
                <button type="button" onClick={() => setLogIndex(GetPrevLogIndex(logIndex))} disabled={logIndex === 0}>Prev</button>
                <button type="button" onClick={() => setLogIndex(GetNextLogIndex(logIndex))} disabled={logIndex === data.length}>Next</button>
                <button type="button" onClick={() => setLogIndex(Array(30).fill(0).reduce((index, _)  => GetNextLogIndex(index), logIndex))} disabled={logIndex === data.length}>+30</button>
            </div>
            <h5><a href={`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/video/`}>{logName}</a></h5>
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