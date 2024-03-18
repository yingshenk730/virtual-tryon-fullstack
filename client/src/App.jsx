import { useState } from 'react'

function App() {
  const [imageSrc, setImageSrc] = useState('')
  const [personImageSrc, setPersonImageSrc] = useState('')
  const [clothImageSrc, setClothImageSrc] = useState('')

  const handlePersonImageUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setPersonImageSrc(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleClothImageUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setClothImageSrc(reader.result)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleImageUpload = async (event) => {
    event.preventDefault()

    const formData = new FormData()
    formData.append('person_image', event.target.person_image.files[0])
    formData.append('cloth_image', event.target.cloth_image.files[0])

    try {
      const response = await fetch('http://127.0.0.1:5000/synthesis-image', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Image synthesis failed')

      const blob = await response.blob()
      const imageSrc = URL.createObjectURL(blob)
      setImageSrc(imageSrc)
    } catch (error) {
      console.error('Error:', error)
    }
  }

  return (
    <div className="w-screen h-screen flex justify-center items-center">
      <div className="w-2/3 h-full bg-slate-100 rounded-xl flex flex-row">
        <form
          onSubmit={handleImageUpload}
          className="flex flex-col gap-3 w-1/2 p-3 px-6">
          <div className="flex flex-col">
            <label htmlFor="person_image">Person Image:</label>
            <input
              type="file"
              id="person_image"
              name="person_image"
              required
              onChange={handlePersonImageUpload}
            />

            {personImageSrc && (
              <img src={personImageSrc} className="w-32" alt="Person Image" />
            )}
          </div>

          <div className="flex flex-col">
            <label htmlFor="cloth_image">Cloth Image:</label>
            <input
              type="file"
              id="cloth_image"
              name="cloth_image"
              required
              onChange={handleClothImageUpload}
            />

            {clothImageSrc && (
              <img src={clothImageSrc} className="w-32" alt="Cloth Image" />
            )}
          </div>

          <button
            className="bg-blue-500 text-white px-4 py-2 rounded-md w-fit"
            type="submit">
            Try On!
          </button>
        </form>
        <div className=" p-6">
          {imageSrc && <img src={imageSrc} className=" w-80" alt="" />}
        </div>
      </div>
    </div>
  )
}

export default App
