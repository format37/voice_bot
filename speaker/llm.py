import os
import asyncio
from openai import OpenAI
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import json
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_api_key(engine):
    
    if engine == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is not set. You need to set the OPENAI_API_KEY environment variable this way:\nexport OPENAI_API_KEY=yourkey")
        return api_key
    
    elif engine == "google":
        # Programmatically get an access token
        creds, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        # Note: the credential lives for 1 hour by default (https://cloud.google.com/docs/authentication/token-types#at-lifetime); after expiration, it must be refreshed.
        return creds.token
    
    else:
        raise ValueError("get_api_key: Invalid engine. Choose between 'openai' and 'google'")


async def llm_request(engine, model, api_key, messages, PROJECT=None, LOCATION=None):
    if engine == "openai":
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content
    
    elif engine == "google":
        client = OpenAI(
        base_url = f'https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT}/locations/{LOCATION}/endpoints/openapi',
        api_key = api_key)
        response = client.chat.completions.create(
            model = f"google/{model}",
            messages = messages
        )

        return response.choices[0].message.content
    else:
        raise ValueError("llm_request: Invalid engine. Choose between 'openai' and 'google'")
    
async def test_openai(messages, model):
    engine = "openai"
    print(f'\n# {engine}')
    
    api_key = get_api_key(engine)
    
    start_time = time.time()
    message = await llm_request(engine, model, api_key, messages)
    end_time = time.time()
    
    return message, end_time - start_time

async def test_google(messages, model):
    engine = "google"
    print(f'\n# {engine}')
    # model = 'gemini-1.5-flash' # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
    PROJECT = 'iceberg-332311' # https://console.cloud.google.com
    LOCATION = 'europe-central2' # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-gemini-using-openai-library
    api_key = get_api_key(engine)
    
    start_time = time.time()
    message = await llm_request(engine, model, api_key, messages, PROJECT, LOCATION)
    end_time = time.time()

    return message, end_time - start_time


def plot_comparison(openai_times, google_times, model_openai, model_google):
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'OpenAI': openai_times,
        'Google': google_times
    })
    df['Test Case'] = range(1, len(openai_times) + 1)
    df_melted = df.melt(id_vars=['Test Case'], var_name='Model', value_name='Time')
    
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Create the main plot
    ax = sns.scatterplot(data=df_melted, x='Test Case', y='Time', hue='Model', style='Model', s=100)
    
    # Add median lines
    for model in ['OpenAI', 'Google']:
        median = np.median(df[model])
        ax.axhline(median, ls='--', color=sns.color_palette()[0 if model == 'OpenAI' else 1], 
                   label=f'{model} Median')
    
    # Fill the area between medians
    openai_median = np.median(df['OpenAI'])
    google_median = np.median(df['Google'])
    ax.fill_between(ax.get_xlim(), openai_median, google_median, alpha=0.2, 
                    color='gray', label='Median Difference')
    
    # Customize the plot
    plt.title(f'Comparison of {model_openai} vs {model_google}')
    plt.xlabel('Test Case')
    plt.ylabel('Time (seconds)')
    plt.legend(title='', loc='upper left', bbox_to_anchor=(1, 1))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('llm_comparison_plot.png')
    plt.close()


async def main():
    # model_openai = "gpt-4o"
    model_openai = "gpt-3.5-turbo-1106"
    model_google = 'gemini-1.5-flash'
    system_content = "Вы - Татьяна, оператор колл центра. Вы звоните клиенту, что бы выяснить оставлял ли он заказ на ремонт. Если оставлял, напишите ЗАЯВКА, если нет, ОШИБКА. Для завершения разговора попрощайтесь с клиентом и напишите hangup. Если это заявка, после завершения разговора клиент будет переведен на оператора для приема заявки."
    user_content = "Алло"
    messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    num_tests = 15 # Default quota is 5 per minute: https://console.cloud.google.com/iam-admin/quotas
    openai_times = []
    google_times = []
    
    for i in range(num_tests):
        openai_message, openai_time = await test_openai(messages, model_openai)
        google_message, google_time = await test_google(messages, model_google)
        openai_times.append(openai_time)
        google_times.append(google_time)
        print(f'[{i} / {num_tests}] OpenAI: {openai_message}')
        print(f'[{i} / {num_tests}] Google: {google_message}')
    
    # Calculate the median run time
    openai_median = np.median(openai_times)
    google_median = np.median(google_times)
    print(f'OpenAI median time: {openai_median}')
    print(f'Google median time: {google_median}')
    
    # Compare the run times. Provide how many times faster
    if openai_median < google_median:
        print(f'OpenAI {model_openai} is {google_median/openai_median:.2f} times faster than Google {model_google}')
    else:
        print(f'Google {model_google} is {openai_median/google_median:.2f} times faster than OpenAI {model_openai}')
    
    # Plot the results
    plot_comparison(openai_times, google_times, model_openai, model_google)


if __name__ == "__main__":
    asyncio.run(main())
