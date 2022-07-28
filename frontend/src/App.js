import { useState } from "react";
import { QueryClient, QueryClientProvider } from "react-query";
import { LogList } from "./LogList.js";
import { LogTimeline } from "./LogTimeline.js";

import './App.css';

const queryClient = new QueryClient();


function App() {
    const [currentLog, setCurrentLog] = useState();

    return (
        <QueryClientProvider client={queryClient}>
            <h1>Robot AI Log Browser</h1>

            <div className="pageContainer">
                <LogList logName={currentLog} onLogSelected={(newLog) => setCurrentLog(newLog)} />
                <LogTimeline logName={currentLog}/>
            </div>
        </QueryClientProvider>
    );
}




export default App;
