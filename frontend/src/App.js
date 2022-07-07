import { QueryClient, QueryClientProvider, useQuery } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";
import axios from "axios";

import './App.css';

const backendUrl = "http://jake-training-box:8000"
const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <LogList />
    </QueryClientProvider>
  );
}

function LogList() {
    const { isLoading, error, data, isFetching } = useQuery("logs", () =>
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
                <li key={log.sha256}>{log.filename}</li>
                )
            }
        </ul>

        <ReactQueryDevtools initialIsOpen />
    </div>
  );
}

export default App;
