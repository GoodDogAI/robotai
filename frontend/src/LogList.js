import { useQuery } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";

import axios from "axios";
import React, { useCallback, useState } from "react";

export function LogList(props) {
  const { onLogSelected, logName } = props;
  const { isLoading, error, data } = useQuery("logs", () =>
    axios.get(
      `${process.env.REACT_APP_BACKEND_URL}/logs`
    ).then((res) => res.data)
  );
  const [openIndex, setOpenIndex] = useState(undefined);

  if (isLoading) return "Loading...";

  if (error) return "An error has occurred: " + error.message;

  if (data.length === 0) return "No logs...";

  return (
    <div className="logList">
      <h4>Available Logs ({data.length})</h4>
      <ul>
        {data.map((logs, index) =>
          <LogListEntry key={index} index={index} logs={logs} isOpen={openIndex === index} selectedLog={logName} onOpen={setOpenIndex} onLogSelected={onLogSelected} />
        )
        }
      </ul>

      <ReactQueryDevtools initialIsOpen />
    </div>
  );
}

function cleanLogName(filename) {
  return "..." + filename.substring(filename.indexOf("-", filename.indexOf("-") + 1) + 1).replace(".log", "");
}

function LogIcon(props) {
  const { validation } = props;
  if (validation === "validatedPassed") return "✅";
  if (validation === "validatedFailed") return "❌";
  return "⚪";
}

function LogListEntry(props) {
  const { logs, index, isOpen, selectedLog, onOpen, onLogSelected } = props;

  const onNameClick = useCallback(() => {
    onOpen(index);
  }, [index, onOpen]);

  const name = logs[0].filename.replace(".log", "");

  return (
    <React.Fragment>
      <li> 
        
        <button className={"link"} onClick={onNameClick}>
          {isOpen ? (<strong>{name}</strong>) : <span>{name}</span> }
        </button>

        {isOpen && (
          <ul>
            {logs.map(log =>
              <li key={log.sha256}>
                <button className={"link"} onClick={() => onLogSelected(log.filename)}>
                  {selectedLog === log.filename ? (<strong>{cleanLogName(log.filename)}</strong>) : <span>{cleanLogName(log.filename)}</span> }
                </button>
                <LogIcon validation={log.meta.validation}/>
              </li>
            )}
          </ul>
        )}

      </li>
    </React.Fragment>
  );
}
