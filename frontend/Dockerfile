# Use Node.js alpine image (lightweight)
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Copy package files first to install dependencies
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend code
COPY . .

# Build the Next.js application for production
RUN npm run build

# Expose the port Next.js runs on
EXPOSE 3000

# Start the application
CMD ["npm", "start"]