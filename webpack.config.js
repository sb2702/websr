const path = require('path');
const webpack = require('webpack');


const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');


module.exports = {
    entry: [ "./src/main.js"],
    output: {
        library: "WebSR",
        libraryTarget: "umd",
        libraryExport: "default",
        path: path.resolve(__dirname, './demo'),
        chunkFilename: 'websr.[name].js',
        filename: "websr.js"
    },
    module: {
        rules: [
            {
                test: /\.js$/,
                exclude: /node_modules/,
                use: {
                    loader: "babel-loader"
                },
            },
        ]
    },
    plugins: [

        new HtmlWebpackPlugin({
            template: 'src/demo/index.html'
        }),

        new CleanWebpackPlugin({
            cleanStaleWebpackAssets: false
        })

    ],


};
