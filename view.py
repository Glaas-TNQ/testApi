from flask import send_from_directory, abort


app.config["CLIENT_IMAGES"] = r"C:\Users\LT_J\OneDrive\Desktop\ML\Faretra\static\uploads"

@app.route("/get_image/<string:image_name>")
def get_image(image_name):

    try:
        return send_from_directory(app.config["CLIENT_IMAGES"], filename=image_name, as_attachment=False) #as_attachment prompts automatic download of the image    except FileNotFoundError:
       
    except:   
        abort(404)