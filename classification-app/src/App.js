import "./App.css";
import React, { useState, useEffect, useFetch } from "react";

function App() {
  const [imageHolder, setImageHolder] = useState(
    "https://imgs.search.brave.com/7tl2Pb9KM6ZC-pdvONJ8pqxdIQIXCPxpWRjV1a09k28/rs:fit:1200:1200:1/g:ce/aHR0cHM6Ly93d3cu/aWNhY2hlZi5jby56/YS93cC1jb250ZW50/L3VwbG9hZHMvMjAx/OS8wMS9JQ0FfUHJv/ZmlsZS1QbGFjZS1I/b2xkZXIucG5n"
  );
  const [data, setData] = useState({});

  let img = null;

  const imageHandler = async (e) => {
    const file = e.target.files[0];
    const reader = new FileReader();
    //console.log(file.name);
    //setImg(file);
    img = file;
    reader.onload = () => {
      if (reader.readyState === 2) {
        const result = reader.result;
        setImageHolder(result);
        // console.log(file.name);

        //console.log(file, "result:", result);
      }
    };
    reader.readAsDataURL(file);
    console.log("llega");
    setData(await postImage());
  };

  let guid = () => {
    let s4 = () => {
      return Math.floor((1 + Math.random()) * 0x10000)
        .toString(16)
        .substring(1);
    };
    //return id of format 'aaaaaaaa'-'aaaa'-'aaaa'-'aaaa'-'aaaaaaaaaaaa'
    return (
      s4() +
      s4() +
      "-" +
      s4() +
      "-" +
      s4() +
      "-" +
      s4() +
      "-" +
      s4() +
      s4() +
      s4()
    );
  };

  async function postImage() {
    const formData = new FormData();
    let name = img.name;
    name = name.replace(/\.[^/.]+$/, ""); // quitar extension
    name = name + guid();
    console.log(name);
    formData.append("img", img);
    formData.append("title", name);
    let headers = new Headers();
    // headers.append("Content-Type", "multipart/form-data");
    headers.append("Accept", "application/json");
    const response = await fetch(
      "http://localhost:8000/api/query/createQuery",
      {
        method: "POST",
        headers: headers,
        body: formData,
      }
    );
    return response.json();
  }
  const dataString = JSON.stringify(data);
  return (
    <div className="page">
      <div className="container">
        <h1 className="heading">Murphy's Magic</h1>
        <div className="image-holder">
          <img src={imageHolder} alt="description" className="img" />
        </div>
        <input
          type="file"
          accept="image/*"
          id="input"
          onChange={imageHandler}
        />
        <div className="label">
          <label htmlFor="input" className="image-upload">
            Seleccione la imagen
          </label>
        </div>
        <div>
          <label className="pred">{dataString}</label>
        </div>
      </div>
    </div>
  );
}

export default App;
