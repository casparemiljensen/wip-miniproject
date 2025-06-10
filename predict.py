import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

def predict_genre(text, model_dir="./bert_genre_model/checkpoint-45", label_encoder_path="label_encoder.pkl"):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    # Load label_encoder
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict (without gradients)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = outputs.logits.argmax(dim=1).item()

    predicted_genre = label_encoder.classes_[pred_id]
    return predicted_genre

if __name__ == "__main__":
    text = "Pulp Fiction is a 1994 American crime film written and directed by Quentin Tarantino, who conceived it with Roger Avary. Starring John Travolta, Samuel L. Jackson, Bruce Willis, Tim Roth, Ving Rhames, and Uma Thurman, it tells several stories of crime in Los Angeles, California. The title refers to the pulp magazines and hardboiled crime novels popular during the mid-20th century, known for their graphic violence and punchy dialogue. Tarantino wrote Pulp Fiction in 1992 and 1993, incorporating scenes that Avary originally wrote for True Romance (1993). Its plot occurs out of chronological order. The film is also self-referential from its opening moments, beginning with a title card that gives two dictionary definitions of pulp. Considerable screen time is devoted to monologues and casual conversations with eclectic dialogue revealing each character's perspectives on several subjects, and the film features an ironic combination of humor and strong violence. TriStar Pictures reportedly turned down the script as too demented. Miramax co-chairman Harvey Weinstein was enthralled, however, and the film became the first that Miramax fully financed. Pulp Fiction won the Palme d'Or at the 1994 Cannes Film Festival, and was a major critical and commercial success. It was nominated for seven awards at the 67th Academy Awards, including Best Picture, and won Best Original Screenplay; it earned Travolta, Jackson, and Thurman Academy Award nominations and boosted their careers. Its development, marketing, distribution, and profitability had a sweeping effect on independent cinema. Pulp Fiction is widely regarded as Tarantino's masterpiece, with particular praise for its screenwriting. The self-reflexivity, unconventional structure, and extensive homage and pastiche have led critics to describe it as a touchstone of postmodern film. It is often considered a cultural watershed, influencing films and other media that adopted elements of its style. The cast was also widely praised, with Travolta, Thurman, and Jackson earning particular acclaim. In 2008, Entertainment Weekly named it the best film since 1983 and it has appeared on many critics' lists of the greatest films ever made. In 2013, Pulp Fiction was selected for preservation in the United States National Film Registry by the Library of Congress as culturally, historically, or aesthetically significant."
    genre = predict_genre(text)
    print(f"Predicted genre: {genre}")
