import { useState } from "react";
import { QueryClient, QueryClientProvider, useQuery } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";
import axios from "axios";

import './App.css';

const backendUrl = "http://jake-training-box:8000"
const queryClient = new QueryClient();



function App() {
    const [currentLog, setCurrentLog] = useState();

    return (
        <QueryClientProvider client={queryClient}>
          <LogList onLogSelected={(newLog) => setCurrentLog(newLog)} />
            <LogTimeline logName={currentLog}/>
        </QueryClientProvider>
    );
}

function LogList(props) {
    const { onLogSelected } = props;
    const { isLoading, error, data } = useQuery("logs", () =>
        axios.get(
          `${backendUrl}/logs`
        ).then((res) => res.data)
      );

  if (isLoading) return "Loading...";

  if (error) return "An error has occurred: " + error.message;

  return (
    <div>
        <h4>Available Logs</h4>
        <ul>
            {data.map(log =>
                <li key={log.sha256}><button onClick={() => onLogSelected(log.filename)}>{log.filename}</button></li>
                )
            }
        </ul>

        <ReactQueryDevtools initialIsOpen />
    </div>
  );
}

function LogTimelineEntry(props) {
    const { data } = props;
    const { valid, logMonoTime, ...rest } = data;

    const mainKey = Object.keys(rest).pop();


    //return (<pre>{JSON.stringify(data, null, 2)}</pre>);
    return (<div>
        {mainKey} @ {logMonoTime}
    </div>);
}

function LogTimeline(props) {
    const { logName } = props;

    const { isLoading, error, data } = useQuery(["logs", logName], () =>
        axios.get(
          `${backendUrl}/logs/${logName}`
        ).then((res) => res.data)
      );

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
        <div>
            { data.map(item => <LogTimelineEntry key={item.logMonoTime} data={item}/>) }

        </div>
    );
}

export default App;
