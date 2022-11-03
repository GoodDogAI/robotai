import { useRef } from "react";
import { useQuery } from "react-query";
import { FixedSizeList as List } from 'react-window';
import classNames from 'classnames';
import axios from "axios";

function parseVecConfig(config) {
    let result = [];

    for (const obs of config) {
        if (obs.type === "msg") {
            let count = 1;

            if (obs.hasOwnProperty("index")) 
                count = Math.abs(obs.index);

            for( let i = 0; i < count; i++) {
                result.push({
                    which: obs.path,
                })
            }
        }
    }

    return result;
}

export function MsgVec(props) {
    const { logName, frameIndex } = props;
    const listEl = useRef(null);
    
    // Load the default brain config and get the msgvec config
    const { data: brainConfig } = useQuery("brainConfig", () => axios.get(`${process.env.REACT_APP_BACKEND_URL}/models/brain/default`).then((res) => res.data));
    const { data: packet } = useQuery(["logs", "msgvec", logName, brainConfig?.basename, frameIndex], () => axios.get(`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/msgvec/${brainConfig?.basename}/${frameIndex}`).then((res) => res.data));

    let obs = [], act = [];
    
    if (brainConfig) {
        obs = parseVecConfig(brainConfig.msgvec.obs);
        act = parseVecConfig(brainConfig.msgvec.act);
    }

    const ObsRow = ({ index, style }) => {
        const rowClass = classNames({
            "row": true,
        });

        let pData = null;

        if (packet) {
            pData = packet.obs[index].toFixed(3);
        }

        return (
            <div className={rowClass} style={style}>
                <div className="cell index">{pData}</div>
                <div className="cell which">{obs[index].which}</div>
            </div>
        );
    }

    const ActRow = ({ index, style }) => {
        const rowClass = classNames({
            "row": true,
        });

        let pData = null;

        if (packet) {
            pData = packet.act[index].toFixed(3);
        }

        return (
            <div className={rowClass} style={style}>
                <div className="cell index">{pData}</div>
                <div className="cell which">{act[index].which}</div>
            </div>
        );
    }

    return (
        <div className="msgvec">
            <div className="logTable">
                <div className="row header">
                    <div className="cell index">Vector</div>
                    <div className="cell which">Observation</div>
                </div>
             <List
                    ref={listEl}
                    height={600}
                    itemCount={obs.length}
                    itemSize={20}
                    width={800}>

                    {ObsRow}
                </List>
            </div>

            <div className="logTable" style={{"marginTop": "3em"}}>
                <div className="row header">
                    <div className="cell index">Vector</div>
                    <div className="cell which">Action</div>
                </div>
             <List
                    height={400}
                    itemCount={act.length}
                    itemSize={20}
                    width={800}>

                    {ActRow}
                </List>
            </div>
        </div>
    );
}