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
    const { isLoading, error, data, isFetching } = useQuery("repoData", () =>
        axios.get(
          `${backendUrl}/logs`
        ).then((res) => res.data)
      );

  if (isLoading) return "Loading...";

  if (error) return "An error has occurred: " + error.message;

  return (
    <div>
      <h1>{data}</h1>
        <ReactQueryDevtools initialIsOpen />
    </div>
  );
}

export default App;
