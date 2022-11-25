import { useState } from "react";
import { useQuery } from "react-query";
import { FixedSizeList as List } from 'react-window';
import classNames from 'classnames';
import axios from "axios";

function parseVecConfig(config) {
    let result = [];
    let baseIndex = 0;

    for (const obs of config) {
        if (obs.type === "msg") {
            let count = 1;

            if (obs.hasOwnProperty("index")) 
                count = Math.abs(obs.index);

            for( let i = 0; i < count; i++) {
                result.push({
                    baseIndex: baseIndex,
                    which: obs.path,
                })
            }

            let timingCount = 0;

            if (obs.hasOwnProperty("timing_index")) 
                timingCount = Math.abs(obs.timing_index);

            for( let i = 0; i < timingCount; i++) {
                result.push({
                    baseIndex: baseIndex,
                    which: "timing",
                })
            }
        }

        baseIndex += 1;
    }

    return result;
}

/*
static float transform_vec_to_msg(const json &transform, float vecValue) {
    const std::string &transformType = transform["type"];

    if (transformType == "identity") {
        return vecValue;
    } else if (transformType == "rescale") {
        const std::vector<float> &msgRange = transform["msg_range"];
        const std::vector<float> &vecRange = transform["vec_range"];

        return std::clamp((vecValue - vecRange[0]) / (vecRange[1] - vecRange[0]) * (msgRange[1] - msgRange[0]) + msgRange[0], msgRange[0], msgRange[1]);
    } else {
        throw std::runtime_error("Unknown transform type: " + transformType);
    }
}*/
function convertVectorToMsg(config, index, value) {
    const obs = config[index];
    console.log(obs);

    if (obs.hasOwnProperty("transform")) {
        const transform = obs.transform;

        if (transform.type === "identity") {
            return value;
        } else if (transform.type === "rescale") {
            const msgRange = transform.msg_range;
            const vecRange = transform.vec_range;

            return Math.min(Math.max((value - vecRange[0]) / (vecRange[1] - vecRange[0]) * (msgRange[1] - msgRange[0]) + msgRange[0], msgRange[0]), msgRange[1]);
        } else {
            throw new Error("Unknown transform type: " + transform.type);
        }
    }

    return value * 10;
}

export function MsgVec(props) {
    const { logName, frameIndex } = props;
    
    // Load the default brain config and get the msgvec config
    const { data: brainConfig } = useQuery("brainConfig", () => axios.get(`${process.env.REACT_APP_BACKEND_URL}/models/brain/default`).then((res) => res.data));
    const { data: packet } = useQuery(["logs", "msgvec", logName, brainConfig?.basename, frameIndex], () => axios.get(`${process.env.REACT_APP_BACKEND_URL}/logs/${logName}/msgvec/${brainConfig?.basename}/${frameIndex}`).then((res) => res.data));

    const [vectorMode, setVectorMode] = useState(true);

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
            if (vectorMode) {
                pData = packet.obs[index].toFixed(3);
            } else {
                pData = convertVectorToMsg(brainConfig.msgvec.obs, obs[index].baseIndex, packet.obs[index]).toFixed(3);
            }
        }

        return (
            <div className={rowClass} style={style}>
                <div className="cell index">{pData}</div>
                <div className="cell which">{obs[index].which}</div>
                <div className="cell index">{index}</div>
            </div>
        );
    }

    const ActRow = ({ index, style }) => {
        const rowClass = classNames({
            "row": true,
        });

        let pData = null;

        if (packet) {
            if (vectorMode) {
                pData = packet.act[index].toFixed(3);
            }
            else {
                pData = convertVectorToMsg(brainConfig.msgvec.act, act[index].baseIndex, packet.act[index]).toFixed(3);
            }
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
            <div className="logTable" onClick={() => setVectorMode(oldVectorMode => !oldVectorMode)}>
                <div className="row header">
                    <div className="cell index">{vectorMode ? "Vector" : "Msg"}</div>
                    <div className="cell which">Observation</div>
                </div>
             <List
                    height={600}
                    itemCount={obs.length}
                    itemSize={20}
                    width={800}>

                    {ObsRow}
                </List>
            </div>

            <div className="logTable" style={{"marginTop": "3em"}}  onClick={() => setVectorMode(oldVectorMode => !oldVectorMode)}>
                <div className="row header">
                    <div className="cell index">{vectorMode ? "Vector" : "Msg"}</div>
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