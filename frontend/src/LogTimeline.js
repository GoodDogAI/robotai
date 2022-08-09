import { useEffect, useState } from "react";
import { useQuery } from "react-query";
import {
    createColumnHelper,
    flexRender,
    getCoreRowModel,
    useReactTable,
  } from '@tanstack/react-table'

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

    const columnHelper = createColumnHelper();
    const columns = [
        columnHelper.accessor('index', {
            accessorFn: (row, index) => index,
        }),
        columnHelper.accessor('which', {
            header: () => <span>type</span>,
            accessorFn: (row, index) => getMessageType(row),
        }),
        columnHelper.accessor('size', {
            header: () => <span>size</span>,
            accessorFn: (row, index) => formatSize(row["_total_size_bytes"]),
        }),
        columnHelper.accessor('json', {
            accessorFn: (row, index) => JSON.stringify(row, null, 2),
            cell: info => {
                if (info.row.index === index) 
                    return (<pre className="messageData">{info.getValue()}</pre>)
            }
        }),
    ];

    const table = useReactTable({
        data,
        columns,
        getCoreRowModel: getCoreRowModel(),
    })


    if (!logName) {
        return <div className="timeline">Select a log to view</div>;
    }

    if (error) {
        return <div className="timeline">Error loading {error}</div>;
    }

    if (isLoading) {
        return <div className="timeline">Loading...</div>;
    }
  
    return (
        <div className="timeline">
            <div className="frameContainer">
                <img width="100%" src={`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/frame/${index}`} alt={`frame${index}`}/>
                <div>
                    <span>Frame {index} / {data.length}</span>
                    <FrameSlider max={data.length - 1} index={index} onChangeIndex={setIndex}/>
                </div>
            </div>
            <h5>{logName}</h5>
            <div className="logTable">
            <table>
                <thead>
                    {table.getHeaderGroups().map((headerGroup) => (
                        <tr key={headerGroup.id}>
                        {headerGroup.headers.map((header) => (
                            <th key={header.id}>
                            {header.isPlaceholder
                                ? null
                                : flexRender(
                                    header.column.columnDef.header,
                                    header.getContext()
                                )}
                            </th>
                        ))}
                        </tr>
                    ))}
                </thead>
                <tbody>
                    {table.getRowModel().rows.map((row) => {
                        return (
                            <tr key={row.id} className={row.index === index ? "selected" : null} onClick={() => setIndex(row.index)}>
                                {row.getVisibleCells().map((cell) => (
                                    <td key={cell.id}>
                                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                                    </td>
                                ))}
                            </tr>);
                    })}
                </tbody>
            </table>
            </div>
        </div>
    );
}