import { useEffect, useState } from "react";
import {  useQuery } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";
import {
    createColumnHelper,
    flexRender,
    getCoreRowModel,
    useReactTable,
  } from '@tanstack/react-table'

import axios from "axios";

function LogTimelineEntry(props) {
    const { data } = props;
    const { valid, logMonoTime, ...rest } = data;

    const mainKey = Object.keys(rest).pop();

    return (<div>
        {mainKey} @ {logMonoTime}
        <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>);
}

function FrameSlider(props) {
    const { index, max, onChangeIndex } = props;

    return (
        <input className="frameSlider" type="range" min="0" max={max} value={index} onChange={(evt) => onChangeIndex(evt.target.value)} />
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

    const [index, setIndex] = useState(0);

    // Reset the frame index back to zero if the logname changes
    useEffect(() => {
        setIndex(0);
    }, [logName]);

    if (!logName) {
        return <div>Select a log to view</div>;
    }

    if (error) {
        return <div>Error loading {error}</div>;
    }

    if (isLoading) {
        return <div>Loading...</div>;
    }

    return (
        <div className="timeline">
            <div className="frameContainer">
                <img width="100%" src={`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/frame/${index}`} alt={`frame${index}`}/>
                <FrameSlider max={data.length - 1} index={index} onChangeIndex={setIndex}/>
            </div>
            <div>
                { data.map(item => <LogTimelineEntry key={item.logMonoTime} data={item}/>) }
            </div>
        </div>
    );
}