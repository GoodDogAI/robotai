import { useQuery } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";

import axios from "axios";

export function LogList(props) {
    const { onLogSelected, logName } = props;
    const { isLoading, error, data } = useQuery("logs", () =>
        axios.get(
          `${process.env.REACT_APP_BACKEND_URL}/logs`
        ).then((res) => res.data)
      );

  if (isLoading) return "Loading...";

  if (error) return "An error has occurred: " + error.message;

  return (
    <div className="logList">
        <h4>Available Logs ({ data.length })</h4>
        <ul>
            {data.map(log =>
                <li key={log.sha256}><button className={log.filename === logName ? "selected" : null} onClick={() => onLogSelected(log.filename)}>{log.filename}</button></li>
                )
            }
        </ul>

        <ReactQueryDevtools initialIsOpen />
    </div>
  );
}
