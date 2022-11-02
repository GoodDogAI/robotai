import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { useQuery } from "react-query";
import { FixedSizeList as List } from 'react-window';
import classNames from 'classnames';
import axios from "axios";

function parseObs(msgvec) {
    let result = [];

    for (const obs of msgvec.obs) {
        if (obs.type === "msg") {
            for( let i = 0; i < Math.abs(obs.index); i++) {
                result.push({
                    which: obs.path,
                })
            }
        }
    }

    return result;
}

export function MsgVec(props) {
    const { logName, index } = props;
    const listEl = useRef(null);
    
    // Load the default brain config and get the msgvec config
    const { data: brainConfig } = useQuery("brainConfig", () => axios.get(`${process.env.REACT_APP_BACKEND_URL}/models/brain/default`).then((res) => res.data));

    let obs = [];
    
    if (brainConfig)
        obs = parseObs(brainConfig.msgvec);

    const Row = ({ index, style }) => {
        const rowClass = classNames({
            "row": true,
        });


        return (
            <div className={rowClass} style={style}>
                <div className="cell index">{index}</div>
                <div className="cell which">{obs[index].which}</div>
            </div>
        );
    }

    console.log(brainConfig);

    return (
        <div className="msgvec">
            <div className="logTable">
             <List
                    ref={listEl}
                    height={1200}
                    itemCount={obs.length}
                    itemSize={20}
                    width={800}>

                    {Row}
                </List>
            </div>
        </div>
    );
}